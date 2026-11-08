"""Focused checks of DAG construction and scalar unary affine semantics."""

import itertools
import unittest

import torch

from fex import (
    DAGBuilder, EliteSearch, Expression, Model, Node, Regression,
    assess, derivatives, evaluate, fit,
)


class DAGBuilderTests(unittest.TestCase):
    def test_shared_dag_choices_and_allocation(self):
        b = DAGBuilder(2)
        x0 = b.input(0)
        offset = b.literal(0.25)
        leaf = b.vsum((None, 'square', 'sin'))
        shared = b.unary(('sin', None), leaf)
        branch = b.binary(('mul', 'sub'), shared, x0)
        squared = b.unary('square', branch, affine=False)
        weight = b.parameter()
        result = b.binary('mul', weight, b.binary('add', squared, shared))
        b.output(b.binary('add', result, offset), shared, branch, shared)

        self.assertEqual(b.action_sizes, (3, 2, 2))
        self.assertEqual(b.parameter_count, 6)  # one vsum, one affine, one parameter
        x = torch.tensor([[-0.6, 0.2], [0.1, 0.9], [0.8, -0.3]], dtype=torch.float64)
        p = torch.linspace(-0.7, 0.8, 6, dtype=x.dtype, requires_grad=True)
        for actions in itertools.product(range(3), range(2), range(2)):
            with self.subTest(actions=actions):
                expr = b.build(actions)
                self.assertEqual(expr, b.build(actions))
                self.assertEqual(expr.parameter_count, 6)
                self.assertEqual(expr.outputs[1], expr.outputs[3])
                self.assertEqual(sum(n.op == 'vsum' for n in expr.nodes), 1)
                self.assertEqual(sum(n.op == 'param' for n in expr.nodes), 3)
                self.assertTrue(all(a < i for i, n in enumerate(expr.nodes) for a in n.args))

                features = (x, x.square(), x.sin())[actions[0]]
                z = (features * p[:2]).sum(dim=1) + p[2]
                h = p[3] * (z.sin() if actions[1] == 0 else z) + p[4]
                branch_value = h * x[:, 0] if actions[2] == 0 else h - x[:, 0]
                expected = torch.stack((
                    p[5] * (branch_value.square() + h) + 0.25, h, branch_value, h,
                ), dim=1)
                actual = evaluate(Model(expr, ()), x, parameters=p)
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(
                    torch.autograd.grad(actual.square().sum(), p)[0],
                    torch.autograd.grad(expected.square().sum(), p)[0],
                )

    def test_affine_values_derivatives_and_coefficient_gradients(self):
        b = DAGBuilder(2)
        leaf = b.vsum('square')
        b.output(b.unary('sin', leaf))
        expr = b.build()
        self.assertEqual(b.action_sizes, ())
        self.assertEqual(expr.parameter_count, 5)
        x = torch.tensor([[-0.4, 0.8], [0.0, 0.0], [0.7, -0.2]], dtype=torch.float64)
        p = torch.tensor([0.6, -0.3, 0.2, 1.7, -0.4], dtype=x.dtype, requires_grad=True)
        actual = derivatives(Model(expr, ()), x, order=2, params=p)
        z = (x.square() * p[:2]).sum(dim=1) + p[2]
        dz = 2 * x * p[:2]
        expected = (
            (p[3] * z.sin() + p[4])[:, None],
            (p[3] * z.cos()[:, None] * dz)[:, None, :],
            (p[3] * (z.cos()[:, None, None] * torch.diag(2 * p[:2])
                     - z.sin()[:, None, None] * dz[:, :, None] * dz[:, None, :]))[:, None],
        )
        for a, e in zip(actual, expected):
            torch.testing.assert_close(a, e, rtol=1e-10, atol=1e-12)
        # Hessian-based objectives must train both leaf and scalar coefficients.
        torch.testing.assert_close(
            torch.autograd.grad(sum(t.square().sum() for t in actual), p)[0],
            torch.autograd.grad(sum(t.square().sum() for t in expected), p)[0],
            rtol=1e-10, atol=1e-12,
        )

    def test_identity_and_per_node_affine_switches(self):
        b = DAGBuilder(1, affine_unary=False)
        x0 = b.input(0)
        plain = b.unary(None, x0)
        scaled = b.unary((None, 'neg'), plain, affine=True)
        squared = b.unary('square', scaled)
        b.output(plain, scaled, b.unary(None, squared, affine=True))
        x = torch.tensor([[-0.2], [0.6]], dtype=torch.float64)
        for action in (0, 1):
            expr = b.build((action,))
            self.assertEqual(expr.parameter_count, 4)
            self.assertEqual(b.action_sizes, (2,))
            h = 2 * (x if action == 0 else -x) + 3
            torch.testing.assert_close(
                evaluate(Model(expr, (2, 3, 4, 5)), x),
                torch.cat((x, h, 4 * h.square() + 5), dim=1),
            )

        # A completely unscaled identity really is the original one-node DAG.
        plain_builder = DAGBuilder(1, affine_unary=False)
        plain_builder.output(plain_builder.unary(None, plain_builder.input(0)))
        self.assertEqual(plain_builder.build(), Expression((Node('input', value=0),), (0,), 1))

    def test_fixed_nodes_lazy_where_and_calculus(self):
        b = DAGBuilder(1)
        x0 = b.input(0)
        positive = b.binary('ge', x0, b.literal(0))
        log_positive = b.unary('log', x0, affine=False)
        log_negative = b.unary('log', b.unary('neg', x0, affine=False), affine=False)
        log_abs = b.node('where', positive, log_positive, log_negative)
        squared = b.unary('square', x0)  # a*x^2 + b
        derivative = b.node('D', squared, value=0)
        integral = b.node('I', squared, value=0)
        b.output(log_abs, derivative, integral)
        x = torch.tensor([[-0.8], [-0.2], [0.4]], dtype=torch.float64)
        actual = evaluate(Model(b.build(), (2.0, 0.3)), x)
        expected = torch.cat((x.abs().log(), 4 * x, 2 * x**3 / 3 + 0.3 * x), dim=1)
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-12)

        # Calculus operators can also occupy a searched scalar unary slot.
        c = DAGBuilder(1)
        child = c.unary('square', c.input(0), affine=False)
        c.output(c.unary(('D', 'I'), child, value=0))
        for action, value in enumerate((2 * x, x**3 / 3)):
            torch.testing.assert_close(
                evaluate(Model(c.build((action,)), (1.4, 0.2)), x),
                1.4 * value + 0.2, rtol=1e-10, atol=1e-12,
            )

    def test_existing_search_and_fit_interfaces(self):
        b = DAGBuilder(1)
        b.output(b.unary(('square', 'sin'), b.input(0)))
        x = torch.linspace(-1, 1, 9, dtype=torch.float64)[:, None]
        objective = Regression(x, 2.5 * x.square() - 0.7)
        result = fit(Model(b.build((0,)), (0.0, 0.0)), objective, steps=20)
        self.assertTrue(result.valid)
        self.assertLess(result.loss, 1e-10)
        search = EliteSearch(b, seed=12)
        expressions = search.ask(4)
        self.assertEqual(expressions, [b.build(a) for a in search.actions.tolist()])
        search.tell([assess(Model(e, (2.5, -0.7)), objective) for e in expressions])
        self.assertIsNone(search.actions)
        self.assertTrue(bool(search.logits.isfinite().all()))

    def test_singleton_slots_and_action_order(self):
        b = DAGBuilder(1, affine_unary=False)
        x0 = b.input(0)

        left = b.unary(('sin',), x0)
        right = b.unary(('square', 'cos'), x0)
        b.output(b.binary(('add',), left, right))

        self.assertEqual(b.action_sizes, (1, 2, 1))

        x = torch.tensor([[-0.2], [0.6]], dtype=torch.float64)

        for choice, op in enumerate((torch.square, torch.cos)):
            expr = b.build((0, choice, 0))
            actual = evaluate(Model(expr, ()), x)
            expected = x.sin() + op(x)
            torch.testing.assert_close(actual, expected)


if __name__ == '__main__':
    unittest.main()
