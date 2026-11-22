"""Known-interface conducting shell with Cartesian-input FEX.

Run one seed:
    python -m examples.conducting_sphere

For ten-seed statistics run:
    python -m benchmarks.conducting_sphere_multiseed

Requires the existing fex package.
No radius or squared-radius feature is supplied to either expression.
"""

from __future__ import annotations

import argparse
import statistics
import math
import time

import torch

from fex import (
    DAGBuilder,
    DomainError,
    EliteSearch,
    FitStage,
    Model,
    assess,
    derivatives,
    evaluate,
    fit,
    fit_stages,
    format_model,
    save_model,
    solve,
)


DTYPE = torch.float64

LEAF = (None, "square", "sin", "cos", "exp")
POST = (None, "square", "sqrt")
BINARY = ("add", "sub", "mul", "div")

N = 48
ROUNDS = 20
BATCH = 8
POOL = 24
COARSE_STEPS = 8
REFINE_STEPS = 20


class Builder(DAGBuilder):
    """Two independent regional trees:

        B(P1(L1(x)), P2(L2(x)))

    Each leaf is:
        L(x) = wx*U(x) + wy*U(y) + wz*U(z) + b

    Coordinate weights are independent.
    """

    def __init__(self):
        # Compatibility: unscaled posts, slots (leaf, post, leaf, post, binary)
        # per region, and four consecutive (wx, wy, wz, b) coefficient blocks.
        super().__init__(3, affine_unary=False)
        outputs = []
        for _ in range(2):
            left = self.unary(POST, self.vsum(LEAF))
            right = self.unary(POST, self.vsum(LEAF))
            outputs.append(self.binary(BINARY, left, right))
        self.output(*outputs)


def initial_model(expression, seed):
    """Unequal random weights with initially positive leaf arguments.

    This is a domain-validity initialization, not an isotropy constraint.
    The bound uses the coordinate box [-2,2]^3, not the norm.
    All coefficients remain unrestricted during fitting.
    """
    generator = torch.Generator().manual_seed(seed)
    p = torch.empty(4, 4, dtype=DTYPE)

    p[:, :3] = 0.2 + 0.8 * torch.rand(
        4, 3, generator=generator, dtype=DTYPE
    )
    p[:, 3] = 0.5 + 2 * p[:, :3].abs().sum(dim=1)

    return Model(expression, tuple(p.flatten().tolist()))


def sample(n, seed, region):
    """Generate xyz coordinates for the supplied spherical geometry."""
    q = torch.quasirandom.SobolEngine(
        3, scramble=True, seed=seed
    ).draw(n, dtype=DTYPE)

    z = 2 * q[:, 0] - 1
    phi = 2 * torch.pi * q[:, 1]
    s = torch.sqrt((1 - z.square()).clamp_min(0))

    direction = torch.stack(
        (s * phi.cos(), s * phi.sin(), z), dim=1
    )

    if region == "inside":
        radius = q[:, 2].pow(1 / 3)
    elif region == "outside":
        radius = (1 + 7 * q[:, 2]).pow(1 / 3)
    elif region == "interface":
        return direction
    elif region == "boundary":
        return 2 * direction
    else:
        raise ValueError(region)

    return radius[:, None] * direction


def dataset(n, seed):
    # Include the origin explicitly. No radial regularity proxy is used.
    inside = torch.cat(
        (
            sample(n, seed, "inside"),
            torch.zeros(1, 3, dtype=DTYPE),
        )
    )

    return (
        inside,
        sample(n, seed + 1, "outside"),
        sample(n, seed + 2, "interface"),
        sample(n, seed + 3, "boundary"),
    )


def blocks(model, parameters, data):
    inside, outside, interface, boundary = data
    ni, no, ng = len(inside), len(outside), len(interface)

    vi, gi, hi = derivatives(
        model.select_outputs((0,)),
        torch.cat((inside, interface)),
        order=2, parameters=parameters,
    )
    vo, go, ho = derivatives(
        model.select_outputs((1,)),
        torch.cat((outside, interface, boundary)),
        order=2, parameters=parameters,
    )

    # Cartesian Laplacian: V_xx + V_yy + V_zz.
    lap_inside = (
        hi[:ni, 0]
        .diagonal(dim1=-2, dim2=-1)
        .sum(dim=-1)
    )
    lap_outside = (
        ho[:no, 0]
        .diagonal(dim1=-2, dim2=-1)
        .sum(dim=-1)
    )

    # On the unit sphere, position equals the outward unit normal.
    normal_jump = (
        (gi[ni:, 0] - go[no:no + ng, 0]) * interface
    ).sum(dim=-1) - 1

    return {
        "pde_inside": lap_inside,
        "pde_outside": lap_outside,
        "continuity": vi[ni:, 0] - vo[no:no + ng, 0],
        "normal_jump": normal_jump,
        "outer_boundary": vo[no + ng:, 0] - 0.5,
    }


def objective(data):
    def call(model, parameters):
        b = blocks(model, parameters, data)

        return (
            b["pde_inside"].square().mean()
            + b["pde_outside"].square().mean()
            + 20 * sum(
                b[name].square().mean()
                for name in (
                    "continuity",
                    "normal_jump",
                    "outer_boundary",
                )
            )
        )

    return call


def optimizer(parameters):
    return torch.optim.LBFGS(
        [parameters],
        max_iter=10,
        max_eval=15,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-12,
        tolerance_change=1e-13,
    )


def relative_l2(model):
    """Independent, volume-weighted error; exact solution used only here."""
    xi = sample(4096, 8000, "inside")
    xo = sample(4096, 8001, "outside")

    with torch.no_grad():
        vi = evaluate(model.select_outputs((0,)), xi)[:, 0]
        vo = evaluate(model.select_outputs((1,)), xo)[:, 0]
        target = 1 / xo.norm(dim=1)

        error = (vi - 1).square().mean() + 7 * (vo - target).square().mean()
        energy = 1 + 7 * target.square().mean()

        return float((error / energy).sqrt())


def main(seed=0, *, rounds=ROUNDS, batch_size=BATCH, pool_size=POOL,
         save_to=None):
    """Search one seed; keep the physics and sampling local to this example."""
    train = objective(dataset(N, 1000))
    validation = objective(dataset(2 * N, 2000))
    start = time.perf_counter()

    # FEX executes the optimizer chain; this problem supplies its settings.
    stages = (
        FitStage(lambda p: torch.optim.Adam([p], lr=0.001), steps=20),
        FitStage(optimizer, steps=COARSE_STEPS),
    )

    def coarse_fit(expression, fit_seed):
        return fit_stages(initial_model(expression, fit_seed), train, stages)

    candidates = solve(
        EliteSearch(Builder(), seed=seed, lr=0.12),
        coarse_fit,
        rounds=rounds,
        batch_size=batch_size,
        pool_size=pool_size,
        seed=seed + 100,
        refine=lambda model, _: fit(
            model, train, optimizer=optimizer, steps=REFINE_STEPS,
        ),
        score=lambda model: assess(model, validation),
    )

    elapsed = time.perf_counter() - start

    if not candidates:
        print(f"{seed:4d}  {'inf':>12}  {'inf':>12}  {elapsed:7.1f}s  no candidate")
        return float("inf")

    best = candidates[0]
    if save_to is not None:
        save_model(best.model, save_to)

    try:
        error = relative_l2(best.model)
        if not math.isfinite(error):
            error = float("inf")
    except DomainError:
        error = float("inf")

    print(
        f"{seed:4d}  {error:12.4e}  {best.loss:12.4e}  "
        f"{elapsed:7.1f}s  "
        f"{format_model(best.model.select_outputs((1,)), variables=('x', 'y', 'z'))[0]}"
    )
    return error

def run(seeds, *, rounds=ROUNDS, batch_size=BATCH, pool_size=POOL,
        save_to=None):
    errors = [
        main(
            seed,
            rounds=rounds,
            batch_size=batch_size,
            pool_size=pool_size,
            save_to=save_to,
        )
        for seed in seeds
    ]

    if len(errors) > 1:
        print(f"\nSuccesses (<1e-5): {sum(e < 1e-5 for e in errors)}/{len(errors)}")
        print(f"Median relative L2: {statistics.median(errors):.4e}")
        print(f"Failed seeds: {[s for s, e in zip(seeds, errors) if e >= 1e-5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=ROUNDS)
    parser.add_argument("--batch-size", type=int, default=BATCH)
    parser.add_argument("--pool-size", type=int, default=POOL)
    parser.add_argument("--save", help="Save the best FEX model as JSON")
    args = parser.parse_args()

    if args.num_seeds < 1:
        parser.error("--num-seeds must be positive")
    if args.save and args.num_seeds != 1:
        parser.error("--save is only supported for a single seed")

    torch.set_num_threads(1)

    print(f"{'Seed':>4}  {'Relative L2':>12}  {'Validation':>12}  {'Time':>8}  Exterior expression")

    run(
        range(args.seed, args.seed + args.num_seeds),
        rounds=args.rounds,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        save_to=args.save,
    )