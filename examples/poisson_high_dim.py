
"""12-D Poisson equation on the unit ball, using vector-to-scalar leaves."""

import torch

from fex import (
    EliteSearch, Expression, Model, Node, Regression,
    assess, derivatives, fit, solve,
)


class Builder:
    """Two independently selected vector leaves with automatic coefficient offsets."""

    operators = ('square', 'sin')

    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        self.action_sizes = (len(self.operators),) * 2

    def build(self, actions):
        nodes = []
        n_parameters = 0

        def add_vsum(transform):
            nonlocal n_parameters
            index = len(nodes)
            nodes.append(Node('vsum', value=n_parameters, transform=transform))
            n_parameters += self.input_dim + 1
            return index

        left = add_vsum(self.operators[actions[0]])
        right = add_vsum(self.operators[actions[1]])
        nodes.append(Node('add', (left, right)))

        return Expression(tuple(nodes), (len(nodes) - 1,),
                          self.input_dim, n_parameters)


def sample_ball(n, d, generator, *, boundary=False):
    x = torch.randn(n, d, dtype=torch.float64, generator=generator)
    x = x / x.norm(dim=1, keepdim=True)
    if not boundary:
        x = x * torch.rand(n, 1, dtype=x.dtype, generator=generator).pow(1 / d)
    return x


def exact(x):
    return 1 - x.square().sum(dim=1, keepdim=True) + 0.25 * x.sin().sum(dim=1, keepdim=True)


def main():
    torch.set_num_threads(1)
    d = 12
    generator = torch.Generator().manual_seed(123)

    interior = sample_ball(48, d, generator)
    boundary_x = sample_ball(64, d, generator, boundary=True)
    boundary = Regression(boundary_x, exact(boundary_x))
    forcing = 2 * d + 0.25 * interior.sin().sum(dim=1, keepdim=True)

    def objective(model: Model, parameters: torch.Tensor):
        _, _, hessian = derivatives(model, interior, order=2, params=parameters)
        laplacian = hessian[:, 0].diagonal(dim1=-2, dim2=-1).sum(dim=1, keepdim=True)
        residual = (-laplacian - forcing).square().mean()
        return residual + 10 * boundary(model, parameters)

    builder = Builder(d)
    search = EliteSearch(builder, seed=3)

    candidates = solve(
        search,
        lambda expr, seed: fit(expr, objective, steps=8, seed=seed),
        rounds=4, batch_size=6, pool_size=3, seed=4,
        refine=lambda model, seed: fit(model, objective, steps=45),
    )
    if not candidates:
        raise RuntimeError('no valid candidate found')

    best = candidates[0]
    test_x = sample_ball(256, d, generator)
    check = Regression(test_x, exact(test_x))

    print('dimension:', d)
    print('operators:', tuple(builder.operators[i] for i in best.actions))
    print('coefficient count:', len(best.model.parameters))
    print('PDE + boundary loss:', best.loss)
    print('independent-point solution MSE:', assess(best.model, check).loss)


if __name__ == '__main__':
    main()