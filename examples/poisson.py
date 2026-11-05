"""Search -u''=2 on [-1,1], u(-1)=u(1)=0. Exact solution: 1-x^2."""
import torch
from dataclasses import dataclass
from fex import EliteSearch, Expression, Node, Regression, Model, assess, fit, solve, derivatives

@dataclass
class Poisson:
    """Mean squared residual of -Laplacian(u)=forcing for a scalar output.

    Inputs and forcing have shapes [N,D] and [N,1], using the fitting dtype and
    device. Build boundary conditions into the expression or the outer objective.
    """

    x: torch.Tensor
    forcing: torch.Tensor

    def __call__(self, model: Model, parameters: torch.Tensor) -> torch.Tensor:
        _, _, hessian = derivatives(model, self.x, order=2, parameters=parameters)
        diagonal = hessian[:, 0].diagonal(dim1=-2, dim2=-1)
        laplacian = diagonal.sum(dim=1, keepdim=True)
        residual = -laplacian - self.forcing
        return residual.square().mean()

class Builder:
    """One selected unary term with a fitted scale and offset."""
    operators = ('identity', 'square', 'cos')
    action_sizes = (len(operators),)

    def build(self, actions):
        nodes = [Node('input', value=0)]
        source = 0
        op = self.operators[actions[0]]
        if op != 'identity':
            nodes.append(Node(op, (0,)))
            source = len(nodes) - 1
        weight = len(nodes)
        nodes.extend((Node('param', value=0),
                      Node('mul', (source, weight)),
                      Node('param', value=1),
                      Node('add', (weight + 1, weight + 2))))
        return Expression(tuple(nodes), (len(nodes) - 1,), 1, 2)


def main():
    torch.set_num_threads(1)
    x = torch.linspace(-1, 1, 33, dtype=torch.float64)[:, None]
    equation = Poisson(x, torch.full_like(x, 2.0))
    boundary_x = torch.tensor([[-1.0], [1.0]], dtype=x.dtype)
    boundary = Regression(boundary_x, torch.zeros_like(boundary_x))

    def objective(model, parameters):
        return equation(model, parameters) + 100 * boundary(model, parameters)

    builder = Builder()
    candidates = solve(EliteSearch(builder, seed=3),
                       lambda expr, seed: fit(expr, objective, steps=20, seed=seed),
                       rounds=3, batch_size=6, seed=4, pool_size=3)
    if not candidates:
        raise RuntimeError('no valid candidate found')
    best = candidates[0]
    test_x = torch.linspace(-1, 1, 201, dtype=torch.float64)[:, None]
    check = Regression(test_x, 1 - test_x.square())
    print('operators:', tuple(builder.operators[a] for a in best.actions))
    print('PDE + boundary loss:', best.loss)
    print('independent-grid solution MSE:', assess(best.model, check).loss)


if __name__ == '__main__':
    main()
