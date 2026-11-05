
"""Verify vectorized elite REINFORCE and Adam against PyTorch autograd."""

import unittest

import torch

from fex import EliteSearch


class ActionSpace:
    """The gradient test only needs action sizes, not expression construction."""

    def __init__(self, sizes):
        self.action_sizes = sizes


class PolicyGradientTests(unittest.TestCase):
    def test_gradient_and_adam_match_autograd(self):
        devices = ['cpu']
        if torch.cuda.is_available():
            devices.append('cuda')

        cases = (
            ((1, 3, 5), 0.0, 0.5),
            ((2, 4, 3, 7), 0.35, 0.5),
            ((8, 2, 1, 5, 3), 0.8, 0.25),
        )
        for device in devices:
            for sizes, epsilon, elite_fraction in cases:
                with self.subTest(device=device, sizes=sizes, exploration=epsilon):
                    self.check_case(device, sizes, epsilon, elite_fraction)

    def check_case(self, device, sizes, epsilon, elite_fraction):
        lr = 0.03
        batch_size = 32
        search = EliteSearch(
            ActionSpace(sizes), seed=17, lr=lr, exploration=epsilon,
            elite_fraction=elite_fraction, device=device,
        )
        with torch.no_grad():
            search.logits.copy_(torch.linspace(
                -1.5, 1.5, search.logits.numel(), dtype=torch.float64, device=device
            ))

        # Independent reference: differentiate the actual log probability of
        # each sampled action under the exploration-mixture policy.
        logits = torch.nn.Parameter(search.logits.detach().clone())
        optimizer = torch.optim.Adam([logits], lr=lr, maximize=True)
        rng = torch.Generator(device=device).manual_seed(123)

        for step in range(5):
            actions = search.ask_actions(batch_size)  # [batch, slots]
            self.assertEqual(tuple(actions.shape), (batch_size, len(sizes)))
            for slot, size in enumerate(sizes):
                self.assertTrue(bool((actions[:, slot] < size).all()))

            rewards = torch.randn(
                batch_size, dtype=torch.float64, device=device, generator=rng
            )
            if step == 2:
                rewards[::3] = 0.0  # Include tied rewards.

            threshold = torch.quantile(
                rewards, 1 - elite_fraction, interpolation='linear'
            )
            weights = (rewards - threshold).clamp_min(0).detach()

            log_prob = torch.zeros_like(rewards)
            for slot, z in enumerate(logits.split(sizes)):
                p = z.softmax(0)
                q = (1 - epsilon) * p + epsilon / len(p)
                log_prob = log_prob + q[actions[:, slot]].log()

            optimizer.zero_grad()
            (weights * log_prob).mean().backward()
            search.tell_rewards(rewards)

            torch.testing.assert_close(
                search.logits.grad, logits.grad, rtol=1e-10, atol=1e-12
            )
            optimizer.step()
            torch.testing.assert_close(
                search.logits, logits, rtol=1e-9, atol=1e-9
            )
            self.assertIsNone(search.actions)


if __name__ == '__main__':
    unittest.main()