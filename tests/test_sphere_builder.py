"""Freeze the supplied sphere grammar; do not run a PDE search or change budgets."""

import itertools
import unittest

import torch

from examples import conducting_sphere as sphere
from fex import Expression, Model, Node, derivatives, evaluate

# Deliberately frozen, independent of the example's configurable vocabularies.
LEAF = (None, "square", "sin", "cos", "exp")
POST = (None, "square", "sqrt")
BINARY = ("add", "sub", "mul", "div")


# Original manual builder, copied verbatim except for the class name.
class LegacyBuilder:
    """Two independent regional trees:

        B(P1(L1(x)), P2(L2(x)))

    Each leaf is:
        L(x) = wx*U(x) + wy*U(y) + wz*U(z) + b

    Coordinate weights are independent.
    """

    action_sizes = (
        len(LEAF),
        len(POST),
        len(LEAF),
        len(POST),
        len(BINARY),
    ) * 2

    def build(self, actions):
        nodes = []
        outputs = []

        for region in range(2):
            u1, p1, u2, p2, binary = map(
                int, actions[5 * region:5 * region + 5]
            )
            children = []

            for side, (u, p) in enumerate(((u1, p1), (u2, p2))):
                index = len(nodes)
                nodes.append(
                    Node(
                        "vsum",
                        value=8 * region + 4 * side,
                        transform=LEAF[u],
                    )
                )

                if POST[p] is not None:
                    nodes.append(Node(POST[p], (index,)))
                    index = len(nodes) - 1

                children.append(index)

            outputs.append(len(nodes))
            nodes.append(Node(BINARY[binary], tuple(children)))

        return Expression(tuple(nodes), tuple(outputs), 3, 16)


class SphereBuilderTests(unittest.TestCase):
    def test_every_regional_action_compiles_to_the_original_dag(self):
        builder, legacy = sphere.Builder(), LegacyBuilder()
        self.assertEqual(builder.action_sizes, (5, 3, 5, 3, 4) * 2)
        fixed = (4, 2, 0, 1, 3)
        # 900 regional combinations, checked in each output position. The
        # regions are independent, so no 810,000-pair enumeration is needed.
        for region in itertools.product(*(range(s) for s in (5, 3, 5, 3, 4))):
            for actions in (region + fixed, fixed + region):
                actual = builder.build(actions)
                self.assertEqual(actual, legacy.build(actions), actions)
                self.assertEqual(actual.parameter_count, 16)
                self.assertEqual([n.value for n in actual.nodes if n.op == 'vsum'],
                                 [0, 4, 8, 12])

    def test_values_physics_loss_and_gradients_are_bitwise_unchanged(self):
        actions = (1, 1, 2, 2, 2, 0, 0, 1, 2, 3)
        actual = sphere.initial_model(sphere.Builder().build(actions), 3)
        expected = sphere.initial_model(LegacyBuilder().build(actions), 3)
        self.assertEqual(actual.parameters, expected.parameters)
        x = torch.tensor([[0.0, 0.0, 0.0], [0.2, -0.3, 0.8], [-0.5, 0.1, -0.9]],
                         dtype=torch.float64)
        p = torch.tensor(actual.parameters, dtype=x.dtype, requires_grad=True)
        torch.testing.assert_close(evaluate(actual, x, parameters=p),
                                   evaluate(expected, x, parameters=p), rtol=0, atol=0)
        for a, e in zip(derivatives(actual, x, order=2, params=p),
                        derivatives(expected, x, order=2, params=p)):
            torch.testing.assert_close(a, e, rtol=0, atol=0)
        objective = sphere.objective(sphere.dataset(4, 1000))
        actual_loss, expected_loss = objective(actual, p), objective(expected, p)
        self.assertTrue(bool(torch.isfinite(actual_loss)))
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        torch.testing.assert_close(torch.autograd.grad(actual_loss, p)[0],
                                   torch.autograd.grad(expected_loss, p)[0], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
