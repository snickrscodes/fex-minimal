
"""Check vector leaves, automatic coefficient offsets, and differentiation."""

import unittest

import torch

from examples.poisson_high_dim import Builder
from fex import Expression, Model, Node, derivatives, evaluate


class VectorLeafTests(unittest.TestCase):
    def test_two_leaves_and_derivatives(self):
        d = 4
        expression = Builder(d).build((0, 1))  # square and sin

        self.assertEqual([n.value for n in expression.nodes[:2]], [0, d + 1])
        self.assertEqual(expression.parameter_count, 2 * (d + 1))

        x = torch.randn(7, d, dtype=torch.float64, requires_grad=True)
        p = torch.linspace(-0.7, 0.8, expression.parameter_count,
                           dtype=x.dtype, requires_grad=True)
        model = Model(expression, tuple(p.detach().tolist()))

        actual = evaluate(model, x, parameters=p)
        expected = (
            (x.square() * p[:d]).sum(dim=1, keepdim=True) + p[d]
            + (x.sin() * p[d + 1:2 * d + 1]).sum(dim=1, keepdim=True) + p[-1]
        )
        torch.testing.assert_close(actual, expected)

        ax, ap = torch.autograd.grad(actual.sum(), (x, p), retain_graph=True)
        ex, ep = torch.autograd.grad(expected.sum(), (x, p), retain_graph=True)
        torch.testing.assert_close(ax, ex)
        torch.testing.assert_close(ap, ep)

        _, jacobian, hessian = derivatives(model, x, order=2, params=p)
        expected_jacobian = 2 * x * p[:d] + x.cos() * p[d + 1:2 * d + 1]
        expected_diagonal = 2 * p[:d] - x.sin() * p[d + 1:2 * d + 1]

        torch.testing.assert_close(jacobian[:, 0], expected_jacobian)
        torch.testing.assert_close(hessian[:, 0], torch.diag_embed(expected_diagonal))

        # A scalar-coordinate leaf can be combined with both vector leaves.
        nodes = expression.nodes + (Node('input', value=2), Node('add', (2, 3)))
        mixed = Expression(nodes, (4,), d, expression.parameter_count)
        torch.testing.assert_close(
            evaluate(Model(mixed, model.parameters), x, parameters=p),
            expected + x[:, 2:3],
        )


if __name__ == '__main__':
    unittest.main()
