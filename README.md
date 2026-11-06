# Minimal FEX

A compact implementation of the **Finite Expression (FEX)** approach: search over discrete mathematical expression structures, fit their continuous coefficients, and reinforce the structures that perform well.

This is a **minimal variant**, not an exact reproduction of the paper's controller. It uses independent, directly trainable categorical logits with elite REINFORCE and Adam instead of a constant-input neural network. The expression grammar and loss functional are supplied by the caller.

## Run

Requires Python 3.10+ and PyTorch. From the project root:

```bash
python -m pip install -e .
python -m examples.poisson
python -m examples.poisson_high_dim
python -m unittest discover -s tests -v
```

The examples demonstrate FEX on two Poisson problems.

The one-dimensional example solves

$$
-u''(x)=2,\qquad x\in[-1,1],\qquad u(-1)=u(1)=0,
$$

whose exact solution is $u(x)=1-x^2$.

The 12-dimensional example solves a Poisson problem on the unit ball with exact solution

$$
u(x)=1-\|x\|_2^2+\frac14\sum_{j=1}^{12}\sin(x_j),
$$

using two vector-to-scalar leaves with automatically allocated coefficient blocks. The forcing and boundary conditions are derived from the exact solution.

Both examples perform expression search, coefficient fitting, and candidate selection, then report the objective loss and solution error on independent evaluation points.

Two focused tests verify the vectorized elite REINFORCE gradient and Adam updates against PyTorch autograd, and the correctness of vector-leaf evaluation, parameter allocation, and differentiation. The policy-gradient test also supports CUDA when available.

## Layout

```text
fex/
    expression.py   expression DAG, evaluation, derivatives, D/I nodes
    leggauss.py     quadrature nodes and weights
    evaluation.py   Regression, fit(), assess(), FitResult, optimizers
    search.py       EliteSearch, Candidate, solve()

examples/
    poisson.py      complete PDE example

tests/
    test_policy_gradient.py
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

It returns a scalar loss. `evaluation.py` includes a weighted-MSE `Regression` convenience objective, which the Poisson example uses to penalize boundary errors and assess its recovered solution. The example defines the Poisson residual objective itself. Other problems can provide their own regression, PDE, or variational objectives without changing FEX.

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

Builders allocate these parameter blocks automatically, advancing the offset for each new leaf and setting `Expression.parameter_count`. Different leaves can also intentionally share coefficients.

Both input types produce scalar values per sample, so they can be combined with the existing scalar operations, derivatives, and integration nodes.

The vector-to-scalar formulation follows the coordinatewise unary reduction used in the original FEX paper, while scalar-coordinate leaves provide additional flexibility.

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

### Examples

Run the original one-dimensional Poisson problem:

```bash
python -m examples.poisson
```

Run a 12-dimensional Poisson problem using two vector leaves:

```bash
python -m examples.poisson_high_dim
```

The high-dimensional example searches for a solution containing a weighted sum of squared coordinates and a weighted sum of their sines. Its builder assigns parameter indices automatically.

### Tests

```bash
python -m unittest discover -s tests -v
```

`test_policy_gradient.py` checks elite REINFORCE against autograd. `test_vsum.py` checks vector-leaf evaluation, coefficient allocation, and differentiation.

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

Input shapes, operator definitions, numerical precision, and valid hyperparameters are largely caller contracts rather than defensive checks. For independent evaluation, use `assess(model, objective)` on data not used by search or refinement. This example and policy-gradient test are smoke checks, not a comprehensive PDE benchmark or proof of correctness.


## Reference

This project is an independent, minimal implementation of the finite expression method introduced in:

Senwei Liang and Haizhao Yang.
[Finite Expression Method for Solving High-Dimensional Partial Differential Equations](https://www.jmlr.org/papers/v26/23-1290.html).
*Journal of Machine Learning Research*, 26(138):1–31, 2025.

[Original implementation](https://github.com/LeungSamWai/Finite-expression-method)

Unlike the original implementation, this project directly optimizes independent categorical logits rather than using a neural-network controller. The implementation is not intended as an exact reproduction of the paper's experiments.
