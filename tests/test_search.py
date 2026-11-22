"""Check the policy update and a complete expression-search call."""

import math
import unittest

import torch

from fex import DAGBuilder, EliteSearch, Regression, fit, solve


class ActionSpace:
    action_sizes = (1, 3, 2)


class SearchTests(unittest.TestCase):
    def test_policy_gradient_matches_autograd(self):
        sizes = ActionSpace.action_sizes
        epsilon = 0.25
        elite_fraction = 0.5
        lr = 0.03

        search = EliteSearch(
            ActionSpace(),
            seed=17,
            lr=lr,
            exploration=epsilon,
            elite_fraction=elite_fraction,
        )

        with torch.no_grad():
            search.logits.copy_(
                torch.linspace(-1, 1, search.logits.numel())
            )

        reference_logits = torch.nn.Parameter(
            search.logits.detach().clone()
        )
        optimizer = torch.optim.Adam(
            [reference_logits], lr=lr, maximize=True
        )
        rewards = torch.linspace(
            -1, 1, 16, dtype=torch.float64
        )

        for _ in range(3):
            actions = search.ask_actions(len(rewards))
            threshold = torch.quantile(
                rewards, 1 - elite_fraction
            )
            weights = (rewards - threshold).clamp_min(0)
            log_prob = torch.zeros_like(rewards)

            for slot, logits in enumerate(
                reference_logits.split(sizes)
            ):
                probabilities = logits.softmax(0)
                behavior = (
                    (1 - epsilon) * probabilities
                    + epsilon / len(probabilities)
                )
                log_prob += behavior[actions[:, slot]].log()

            optimizer.zero_grad()
            (weights * log_prob).mean().backward()
            search.tell_rewards(rewards)

            torch.testing.assert_close(
                search.logits.grad,
                reference_logits.grad,
                rtol=1e-10,
                atol=1e-12,
            )

            optimizer.step()
            torch.testing.assert_close(
                search.logits,
                reference_logits,
                rtol=1e-9,
                atol=1e-9,
            )
            self.assertIsNone(search.actions)

    def test_solve_fits_and_ranks_candidates(self):
        builder = DAGBuilder(1)
        builder.output(
            builder.unary(
                ("square", "sin"), builder.input(0)
            )
        )

        x = torch.linspace(
            -1, 1, 13, dtype=torch.float64
        )[:, None]
        objective = Regression(
            x, 1.7 * x.square() + 0.3
        )

        candidates = solve(
            EliteSearch(builder, seed=7),
            lambda expression, seed: fit(
                expression, objective, steps=15, seed=seed
            ),
            rounds=2,
            batch_size=6,
            pool_size=2,
            seed=17,
        )

        self.assertTrue(candidates)
        self.assertTrue(
            all(math.isfinite(c.loss) for c in candidates)
        )
        losses = [c.loss for c in candidates]
        self.assertEqual(losses, sorted(losses))
        self.assertLess(candidates[0].loss, 1e-4)


if __name__ == "__main__":
    unittest.main()
