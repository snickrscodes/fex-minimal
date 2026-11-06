from collections.abc import Callable, Sequence
from dataclasses import dataclass
import random

import torch
from torch import Tensor

from .evaluation import FitResult
from .expression import Expression, Model

Fitter = Callable[[Expression, int], FitResult]
Refiner = Callable[[Model, int], FitResult]
Scorer = Callable[[Model], FitResult]


@dataclass(frozen=True)
class Candidate:
    actions: tuple[int, ...]
    model: Model
    loss: float


def loss_reward(loss: Tensor) -> Tensor:
    return 1 / (1 + torch.sqrt(loss.clamp(max=200000)))

class EliteSearch:
    def __init__(self, 
        builder, *, seed: int = 0, lr: float = 0.1,
        exploration: float = 0.1, elite_fraction: float = 0.5,
        reward: Callable[[Tensor], Tensor] = loss_reward,
        invalid_reward: float = 0.0,
        device: torch.device | str = 'cpu'
    ):

        self.action_sizes = tuple(builder.action_sizes)
        self.builder = builder
        self.exploration, self.elite_fraction = exploration, elite_fraction
        self.reward, self.invalid_reward = reward, invalid_reward

        device = torch.device(device)
        sizes = torch.tensor(self.action_sizes, dtype=torch.long, device=device)

        # pad categorical distributions to (slots, max_options)
        # invalid positions are never sampled nor included in gradients
        self._mask = torch.arange(max(self.action_sizes), device=device)[None, :] < sizes[:, None]
        self._uniform = self._mask.to(torch.float64) / sizes[:, None]

        self.rng = torch.Generator(device=device).manual_seed(seed)

        # directly optimize logits instead, fex paper just uses a NN with constant input
        self.logits = torch.nn.Parameter(torch.zeros(sum(self.action_sizes), dtype=torch.float64, device=device))
        self._optimizer = torch.optim.Adam([self.logits], lr=lr, maximize=True)
        self._batch = None

    @property
    def actions(self) -> Tensor | None:
        return None if self._batch is None else self._batch[0]

    def _padded_probabilities(self) -> Tensor:
        padded = self.logits.detach().new_full(self._mask.shape, -torch.inf)
        padded.masked_scatter_(self._mask, self.logits.detach())
        return padded.softmax(dim=1)

    def probabilities(self) -> tuple[Tensor, ...]:
        return self._padded_probabilities()[self._mask].split(self.action_sizes)

    def ask_actions(self, batch_size: int) -> Tensor:
        if self._batch is not None:
            raise RuntimeError('finish the pending batch with tell before ask')

        epsilon = self.exploration
        policies = self._padded_probabilities()
        behavior = (1 - epsilon) * policies + epsilon * self._uniform

        actions = torch.multinomial(behavior, batch_size, replacement=True, generator=self.rng).T.contiguous()

        self._batch = actions, policies, behavior, epsilon
        return actions

    def ask(self, batch_size: int) -> list[Expression]:
        actions = self.ask_actions(batch_size)
        return [self.builder.build(tuple(row)) for row in actions.tolist()]

    def tell(self, fits: Sequence[FitResult]) -> None:
        device = self.logits.device
        losses = torch.tensor([f.loss for f in fits], dtype=torch.float64, device=device)
        valid = torch.tensor([f.valid for f in fits], dtype=torch.bool, device=device)

        if not bool(torch.isfinite(losses[valid]).all()):
            raise ValueError('valid fit has nonfinite loss')

        rewards = torch.full_like(losses, self.invalid_reward)
        rewards[valid] = self.reward(losses[valid])
        self.tell_rewards(rewards)

    def tell_rewards(self, rewards: Tensor) -> None:
        if self._batch is None:
            raise RuntimeError('ask must precede tell')

        actions, policies, behavior, epsilon = self._batch
        rewards = torch.as_tensor(rewards, dtype=torch.float64, device=self.logits.device)

        threshold = torch.quantile(rewards, 1 - self.elite_fraction, interpolation='linear')
        weights = (rewards - threshold).clamp_min(0)

        # exact score-function gradient
        selected = actions.T.contiguous()
        credit = weights[None, :] * (1 - epsilon) * policies.gather(1, selected) / behavior.gather(1, selected)

        received = torch.zeros_like(policies).scatter_add_(1, selected, credit)
        gradient = ((received - credit.sum(1, keepdim=True) * policies) / len(actions))[self._mask]

        self.logits.grad = gradient
        self._optimizer.step()
        self._batch = None


def solve(search: EliteSearch, fit_candidate: Fitter, *, rounds: int,
    batch_size: int = 10, pool_size: int = 10, seed: int = 0,
    refine: Refiner | None = None, score: Scorer | None = None,
) -> tuple[Candidate, ...]:

    # fit_candidate own the optimizer and schedule, returns fitresult
    # same with refine, but continues a fit
    # optional score function replaces the fitting loss for BOTH pool ranking and policy gradient

    rng = random.Random(seed)
    pool: dict[tuple[int, ...], Candidate] = {}

    def measure(result: FitResult):
        if result.valid and score is not None:
            result = score(result.model)
        return result

    def ranked(): # sort candidates ascending by loss
        return tuple(sorted(pool.values(), key=lambda c: c.loss))

    for _ in range(rounds):
        expressions = search.ask(batch_size)
        actions = [tuple(row) for row in search.actions.tolist()] # save before tell(), which may clear the pending batch
        fits = [measure(fit_candidate(expression, rng.randrange(2**63))) for expression in expressions]
        search.tell(fits)

        for action, result in zip(actions, fits):
            if result.valid and (action not in pool or result.loss < pool[action].loss):
                pool[action] = Candidate(action, result.model, result.loss)

        top = ranked()[:pool_size] # top k, discarded structures can re-enter through a better subsequent fit
        pool = {c.actions: c for c in top}

    if refine is not None:
        for candidate in ranked():
            result = measure(refine(candidate.model, rng.randrange(2**63)))
            if result.valid and result.loss < candidate.loss:
                pool[candidate.actions] = Candidate(candidate.actions, result.model, result.loss)

    return ranked()