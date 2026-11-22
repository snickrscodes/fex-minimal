"""Core expression DAG, vector leaves, and calculus checks."""

import unittest

import torch

from fex import DAGBuilder, DomainError, Model, derivatives, evaluate


class ExpressionTests(unittest.TestCase):
    def test_vector_leaves_and_derivatives(self):
        d = 3
        builder = DAGBuilder(d, affine_unary=False)
        left = builder.vsum(("square", "sin"))
        right = builder.vsum(("square", "sin"))
        builder.output(builder.binary("add", left, right))
        expression = builder.build((0, 1))

        self.assertEqual(builder.action_sizes, (2, 2))
        self.assertEqual(expression.parameter_count, 2 * (d + 1))

        x = torch.tensor(
            [[-0.5, 0.2, 0.7], [0.3, -0.2, 0.8]],
            dtype=torch.float64,
        )
        p = torch.linspace(
            -0.5, 1.0, 2 * (d + 1),
            dtype=x.dtype, requires_grad=True,
        )
        model = Model(expression, tuple(p.detach().tolist()))

        expected = (
            (x.square() * p[:d]).sum(dim=1, keepdim=True) + p[d]
            + (x.sin() * p[d + 1:2 * d + 1]).sum(dim=1, keepdim=True)
            + p[-1]
        )
        actual, jacobian, hessian = derivatives(
            model, x, order=2, parameters=p
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            jacobian[:, 0],
            2 * x * p[:d] + x.cos() * p[d + 1:2 * d + 1],
        )
        torch.testing.assert_close(
            hessian[:, 0],
            torch.diag_embed(
                2 * p[:d] - x.sin() * p[d + 1:2 * d + 1]
            ),
        )
        torch.testing.assert_close(
            torch.autograd.grad(
                actual.square().sum(), p, retain_graph=True
            )[0],
            torch.autograd.grad(expected.square().sum(), p)[0],
        )

    def test_shared_dag_and_choices(self):
        builder = DAGBuilder(2, affine_unary=False)
        leaf = builder.vsum(("square", "sin"))
        x0 = builder.input(0)
        result = builder.binary(("add", "sub"), leaf, x0)
        builder.output(result, leaf, leaf)

        x = torch.tensor(
            [[0.2, -0.3], [0.7, 0.4]], dtype=torch.float64
        )
        p = (0.5, -0.25, 0.1)

        for a in range(2):
            for b in range(2):
                with self.subTest(actions=(a, b)):
                    expression = builder.build((a, b))
                    self.assertEqual(
                        expression.outputs[1],
                        expression.outputs[2],
                    )

                    leaf_value = (
                        (x.square() if a == 0 else x.sin())
                        * torch.tensor(p[:2], dtype=x.dtype)
                    ).sum(dim=1, keepdim=True) + p[2]

                    first = (
                        leaf_value + x[:, :1]
                        if b == 0 else leaf_value - x[:, :1]
                    )
                    expected = torch.cat(
                        (first, leaf_value, leaf_value), dim=1
                    )

                    torch.testing.assert_close(
                        evaluate(Model(expression, p), x),
                        expected,
                    )

    def test_lazy_where_and_calculus(self):
        b = DAGBuilder(1, affine_unary=False)
        x = b.input(0)
        condition = b.binary("ge", x, b.literal(0))
        positive = b.unary("log", x)
        negative = b.unary("log", b.unary("neg", x))
        b.output(b.node("where", condition, positive, negative))

        points = torch.tensor(
            [[-0.8], [-0.2], [0.4]], dtype=torch.float64
        )
        torch.testing.assert_close(
            evaluate(Model(b.build(), ()), points),
            points.abs().log(),
        )

        calculus = DAGBuilder(1, affine_unary=False)
        x = calculus.input(0)
        square = calculus.unary("square", x)
        calculus.output(
            calculus.node("D", square, value=0),
            calculus.node("I", square, value=0),
        )

        expected = torch.cat(
            (2 * points, points.pow(3) / 3), dim=1
        )
        torch.testing.assert_close(
            evaluate(Model(calculus.build(), ()), points),
            expected, rtol=1e-10, atol=1e-12,
        )

    def test_sqrt_domain(self):
        b = DAGBuilder(1, affine_unary=False)
        b.output(b.unary("sqrt", b.input(0)))
        model = Model(b.build(), ())

        x = torch.tensor(
            [[0.25], [4.0]], dtype=torch.float64
        )
        values, jacobian = derivatives(model, x)

        torch.testing.assert_close(values, x.sqrt())
        torch.testing.assert_close(
            jacobian[:, 0, 0],
            0.5 / x.sqrt()[:, 0],
        )

        with self.assertRaises(DomainError):
            evaluate(model, -x)


if __name__ == "__main__":
    unittest.main()
