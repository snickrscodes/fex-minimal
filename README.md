# Minimal FEX

A compact implementation of the **Finite Expression (FEX)** approach: search over discrete mathematical expression structures, fit their continuous coefficients, and reinforce the structures that perform well.

This is a **minimal variant**, not an exact reproduction of the paper's controller. It uses independent, directly trainable categorical logits with elite REINFORCE and Adam instead of a constant-input neural network. The expression grammar and loss functional are supplied by the caller.

## Run

Requires Python 3.10+ and PyTorch. From the project root:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Run the conducting-sphere example with a single seed:

```bash
python -m examples.conducting_sphere
```

For a quick smoke test, reduce the search budget:

```bash
python -m examples.conducting_sphere --rounds 2 --batch-size 4 --pool-size 4
```

To reproduce the original ten-seed experiment:

```bash
python -m examples.conducting_sphere --num-seeds 10
```

The `--seed` argument specifies the starting seed (default 0). The default number of seeds is 1. A single-seed model can be saved using `--save model.json`.

## Examples

### Conducting-sphere interface

`examples/conducting_sphere.py` searches for separate interior and exterior potentials using only Cartesian inputs $(x,y,z)$. The known interface is the unit sphere, with an outer boundary at radius 2. The equations are

$$
\Delta u_{\mathrm{in}}=0\quad (\|x\|_2<1),\qquad
\Delta u_{\mathrm{out}}=0\quad (1<\|x\|_2<2),
$$

with continuity, a normal-derivative jump, and a prescribed outer value:

$$
u_{\mathrm{in}}=u_{\mathrm{out}},\qquad
n\cdot(\nabla u_{\mathrm{in}}-\nabla u_{\mathrm{out}})=1
\quad (\|x\|_2=1),\qquad
u_{\mathrm{out}}=\frac12\quad (\|x\|_2=2).
$$

The exact potentials, used only for independent error assessment, are

$$
u_{\mathrm{in}}(x)=1,\qquad u_{\mathrm{out}}(x)=\frac1{\|x\|_2}.
$$

Each regional expression combines two vector leaves, optional unscaled unary post-operators, and a binary operator. `DAGBuilder` provides ten categorical action slots and 16 trainable coefficients across both outputs. Neither radius nor squared radius is supplied as an input feature. The strict scalar `sqrt` operator is built into FEX; invalid expression domains are rejected rather than clipped.

Each seed uses domain-aware coefficient initialization, a reusable `fit_stages()` schedule (Adam followed by L-BFGS), pool refinement, and search-time scoring on separate validation points.

The example reports validation loss, volume-weighted relative \(L^2\) error on independent samples, runtime, and the fitted exterior expression. A single-seed run can save its fitted model as JSON. The optional ten-seed run additionally reports success counts, median relative error, and failed seeds.

**Validation loss is used for search feedback and ranking**, so the independent relative-error points are the held-out check.

## Layout

```text
fex/
    __init__.py         public API
    expression.py       expression DAG, DAGBuilder, evaluation, derivatives, D/I nodes
    leggauss.py         quadrature nodes and weights
    evaluation.py       Regression, fit(), fit_stages(), assess(), optimizers
    search.py           EliteSearch, Candidate, solve()
    formatting.py       format_model() for readable mathematical expressions
    serialization.py    save_model(), load_model() JSON format

examples/
    conducting_sphere.py   3-D conducting-sphere interface problem (supports single or multi-seed)

benchmarks/
    conducting_sphere_multiseed.py   ten-seed interface experiment

tests/
    test_builder.py            DAG construction and scalar operators
    test_policy_gradient.py    elite REINFORCE gradient and Adam updates
    test_sphere_builder.py     sphere grammar and physics-loss compatibility
    test_vsum.py               vector leaves and differentiation
    test_public_api.py         selection, sqrt, staged fitting, formatting, JSON
```

## Search space and objective

A **builder** specifies which expression structures can be sampled:

- `action_sizes`: number of choices at each discrete action slot; e.g. `(3, 2)` means three choices for one slot and two for another.
- `build(actions) -> Expression`: translates the sampled integer choices into an expression.

`EliteSearch` does not know about unary or binary operators, tree topology, or the meanings of action indices. Those belong to the builder. It samples a tensor of shape `(batch_size, num_slots)`; `ask()` turns its rows into `Expression` objects using the builder.

An objective is a differentiable callable:

```python
objective(model: Model, parameters: torch.Tensor) -> torch.Tensor
```

It returns a scalar loss. `evaluation.py` includes a weighted-MSE `Regression` convenience objective. The PDE residual objectives are defined in the examples themselves. Other problems can provide their own regression, PDE, or variational objectives without changing FEX.

### Expression inputs

Expressions support two kinds of input leaves:

- `Node('input', value=i)` selects the scalar coordinate `x_i`.
- `Node('vsum', value=s, transform=op)` applies a unary operator elementwise to the full input vector, then computes a learned weighted sum with a bias.

For $x \in \mathbb{R}^D$, a vector leaf computes

$$
h(x) = \sum_{j=0}^{D-1} \alpha_j \phi(x_j) + \beta.
$$

Here, `transform=op` selects the unary function from `OPERATIONS`, while `transform=None` uses the identity.

The `value=s` field specifies the starting coefficient index. Each vector leaf consumes `D + 1` parameters:

- `parameters[s:s+D]`: coordinate weights
- `parameters[s+D]`: bias

For example:

```python
expression = Expression(
    nodes=(Node('vsum', value=0, transform='square'),),
    outputs=(0,),
    input_dim=50,
    parameter_count=51,
)
```

`DAGBuilder.vsum()` allocates these parameter blocks automatically, advancing the offset for each new leaf and setting `Expression.parameter_count`. Custom builders can also track offsets explicitly if necessary. Expressions constructed directly can also intentionally share coefficients between leaves.

Both input types produce scalar values per sample, so they can be combined with the existing scalar operations, derivatives, and integration nodes.

The vector-to-scalar formulation follows the coordinatewise unary reduction used in the original FEX paper, while scalar-coordinate leaves provide additional flexibility.

### DAG construction

`DAGBuilder` constructs `Expression` objects with automatic node indexing, coefficient allocation, and categorical operator choices. It supports arbitrary DAGs, shared subexpressions, and multiple outputs.

```python
from fex import DAGBuilder

b = DAGBuilder(input_dim=3)

leaf = b.vsum(("square", "sin"))
root = b.unary(("cos", None), leaf)
b.output(root)

print(b.action_sizes)     # (2, 2)
print(b.parameter_count)  # 6

expression = b.build((0, 1))
```

A string (or `None` for identity) specifies a fixed unary operator; a sequence specifies categorical choices. Action slots follow declaration order. `build(actions)` resolves these choices into an ordinary `Expression` DAG.

Vector leaves compute

$$
L(x)=\sum_{j=1}^D w_j\phi(x_j)+b.
$$

Scalar unary operations default to

$$
U(z)=\alpha\phi(z)+\beta.
$$

Set `affine_unary=False` on the builder, or `affine=False` on an individual unary operation, to omit the scale and bias. Reusing a builder handle shares the corresponding computation and parameters.

The builder is optional; `Node` and `Expression` can still be constructed directly.

## Fitting and search

Given a builder and objective, the core workflow is:

```python
from fex import EliteSearch, fit, solve

search = EliteSearch(builder, seed=0)  # builder and objective defined by caller

candidates = solve(
    search,
    lambda expression, seed: fit(expression, objective, steps=100, seed=seed),
    rounds=100,
    batch_size=10,
    pool_size=10,
)

if candidates:
    best = candidates[0]  # lowest measured loss
    print(best.actions, best.model.parameters, best.loss)
```

`fit(Expression, ...)` initializes its coefficients from a local seeded generator. `fit(Model, ...)` starts from the model's existing coefficients, which makes it suitable for optional warm-started refinement. It returns a `FitResult(model, loss, valid)` and retains the best finite coefficient snapshot encountered. An optimizer domain failure terminates that attempt while preserving any valid incumbent; an invalid initial point has no incumbent to preserve. Programming errors other than `DomainError` propagate.

`solve()` calls `search.ask()`, fits each proposed expression, calls `search.tell()`, and retains up to `pool_size` distinct action tuples with their best measured fitted models. Repeated structures are fitted independently. Optional `refine(model, seed)` continues the retained fits. Optional `score(model) -> FitResult` replaces the fitting loss for **both** search feedback and pool ranking, including acceptance of refined results. Scoring data is therefore part of search, not a held-out test set.

`solve()` returns a tuple of `Candidate(actions, model, loss)` in ascending loss order, or `()` when no candidate was valid. There is no separate result wrapper, progress callback, or checkpoint API.

## Reusable fitting, outputs, and model artifacts

A numerical fitting schedule belongs to FEX, but the solver choice, number of steps, initial parameters, and objective remain caller-specified. `FitStage` and `fit_stages()` run one fresh optimizer per stage, warm-started from the previous stage's best valid model. If a later stage fails, the last valid incumbent is preserved. An initially invalid model remains invalid.

```python
from fex import FitStage, fit_stages, format_model, save_model, load_model
import torch

# start is an Expression or an initialized Model; objective is caller-defined.
stages = (
    FitStage(lambda p: torch.optim.Adam([p], lr=0.001), steps=20),
    FitStage(lambda p: torch.optim.LBFGS(
        [p], max_iter=10, max_eval=15, line_search_fn='strong_wolfe'
    ), steps=8),
)
result = fit_stages(start, objective, stages, seed=0)
if result.valid:
    # An output view shares the original DAG and coefficients, but evaluates
    # only the selected output(s), useful for piecewise regional solutions.
    regional_model = result.model.select_outputs((0,))
    print(format_model(regional_model, variables=('x', 'y', 'z'))[0])
    save_model(result.model, 'model.json')
    restored_model = load_model('model.json')
```

`format_model()` always returns a tuple of human-readable expressions; these strings are *not* executable code. `save_model()`/`load_model()` preserve a versioned JSON representation of the DAG and fitted coefficient vector. Custom operator names are serialized but their Python implementations are not; register such operators before evaluating a restored model. They should not be used as a substitute for the caller's PDE, sampling, or objective code.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover DAG construction, affine and unscaled scalar operations, lazy branching, derivatives and integration (`test_builder.py`); elite REINFORCE gradients and Adam updates against PyTorch autograd (`test_policy_gradient.py`); equivalence of the conducting-sphere grammar to its original hand-built DAG and its physics-loss gradients (`test_sphere_builder.py`); vector-leaf evaluation, coefficient allocation, and differentiation (`test_vsum.py`); and public API behavior (`test_public_api.py`). The policy-gradient test also supports CUDA when available.

## Policy update

Each slot has trainable logits $z_i$ and learned probabilities $p_i=\mathrm{softmax}(z_i)$. For a slot with $K_i$ options, actions are sampled from

$$
q_i(a)=(1-\epsilon)p_i(a)+\frac{\epsilon}{K_i},
$$

where $\epsilon$ is the exploration fraction. The joint distribution factorizes across slots.

After fitting a batch of $B$ expressions, the default reward is

$$
R(L) = \frac{1}{1+\sqrt{\min(L,200000)}},
$$

following the reward transformation used in the authors' public implementation, with a loss cap of 200000. This assumes nonnegative objective losses. Invalid fits receive a reward of zero.

The elite weights are

$$
\tau=\mathrm{quantile}_{1-\rho}(R_1,\ldots,R_B),
\qquad w_b=\max(R_b-\tau,0),
$$

where $\rho$ is `elite_fraction`. The controller uses the score-function gradient

$$
\frac1B\sum_{b=1}^{B}w_b\nabla_z\log q(a_b).
$$

It computes this gradient analytically using padded categorical tensors and `scatter_add_`, then supplies it to Adam with `maximize=True`. Padding never contributes an action or a trainable logit. The logits, sampling RNG, and update tensors can run on CPU or CUDA via `EliteSearch(..., device='cuda')`; coefficient fitting has its own `device` argument.

The controller is **not autoregressive**: operator decisions are independent given its current logits. This keeps the method small and fast, but does not model conditional dependencies between operator choices. Its direct-logit controller and elite-weighting convention intentionally differ from the paper's neural-network implementation.

## Numerical notes

`expression.py` evaluates symbolic DAGs with PyTorch operations, including strict/protected operators, lazy `where`, derivatives, and numerical integration (`D`/`I` nodes). The `I` node uses 16-point quadrature, and differentiating it differentiates the numerical rule. `Regression` supports optional sample weights and chunking; the weighted loss normalizes weights to sum to one.

Input shapes, operator definitions, numerical precision, and valid hyperparameters are largely caller contracts rather than defensive checks. For independent evaluation, use `assess(model, objective)` on data not used by search, scoring, or refinement. The bundled example and focused tests are demonstrations and smoke checks, not a comprehensive PDE benchmark or proof of correctness.


## Reference

This project is an independent, minimal implementation of the finite expression method introduced in:

Senwei Liang and Haizhao Yang.
[Finite Expression Method for Solving High-Dimensional Partial Differential Equations](https://www.jmlr.org/papers/v26/23-1290.html).
*Journal of Machine Learning Research*, 26(138):1–31, 2025.

[Original implementation](https://github.com/LeungSamWai/Finite-expression-method)

Unlike the original implementation, this project directly optimizes independent categorical logits rather than using a neural-network controller. The implementation is not intended as an exact reproduction of the paper's experiments.
