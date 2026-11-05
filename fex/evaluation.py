from collections.abc import Callable
from dataclasses import dataclass
import math

import torch
from torch import Tensor

from .expression import DomainError, Expression, Model, evaluate

@dataclass
class Regression:
    x: Tensor # (N, D)
    y: Tensor # (N, 0)
    weights: Tensor | None = None
    chunk_size: int | None = None

    def __call__(self, model: Model, parameters: Tensor) -> Tensor:

        weights = self.weights
        if weights is not None:
            # scale first to prevent overflow
            scaled = weights.double() / weights.max()
            weights = scaled / scaled.sum()

        total = (parameters * 0).sum().double() # anchor (evil trick) for autograd graph connection (in case of D nodes)

        N, O = self.y.shape
        size = N if self.chunk_size is None else self.chunk_size

        for begin in range(0, N, size):
            end = begin + size

            y_hat = evaluate(model, self.x[begin:end], parameters=parameters)
            r = y_hat.double() - self.y[begin:end].double()
            err = r * r

            if weights is not None:
                err = err * weights[begin:end, None]
                total = total + err.sum() / O
            else:
                total = total + err.sum() / (N * O)
        return total


Objective = Callable[[Model, Tensor], Tensor]

def adam(parameters: Tensor) -> torch.optim.Optimizer:
    return torch.optim.Adam([parameters], lr=0.05)

def lbfgs(parameters: Tensor) -> torch.optim.Optimizer:
    return torch.optim.LBFGS([parameters], max_iter=1, max_eval=10, line_search_fn='strong_wolfe')

@dataclass(frozen=True)
class FitResult:
    model: Model
    loss: float
    valid: bool


def assess(model: Model, objective: Objective, *, dtype=torch.float64, device='cpu') -> FitResult:
    parameters = torch.tensor(model.parameters, dtype=dtype, device=device, requires_grad=True)
    try:
        loss = objective(model, parameters)
        value = float(loss.detach())
        if not math.isfinite(value):
            raise DomainError('nonfinite objective')
        return FitResult(model, value, True)
    except DomainError:
        return FitResult(model, math.inf, False)


def fit(start: Expression | Model, objective: Objective, optimizer: Callable[[Tensor], torch.optim.Optimizer] = lbfgs,
        *, steps=100, seed=0, dtype=torch.float64, device='cpu') -> FitResult:
    # seeded random coefficients, models retain theirs
    if isinstance(start, Expression):
        generator = torch.Generator().manual_seed(seed)
        initial = torch.randn(start.parameter_count, generator=generator, dtype=dtype)
        start = Model(start, tuple(initial.tolist()))

    parameters = torch.tensor(start.parameters, dtype=dtype, device=device, requires_grad=True)
    best = assess(start, objective, dtype=dtype, device=device)
    if not best.valid or parameters.numel() == 0 or steps == 0:
        return best

    solver = optimizer(parameters)

    def remember(loss):
        nonlocal best
        value = float(loss.detach())
        if not math.isfinite(value):
            raise DomainError('nonfinite fitting objective')
        if value < best.loss:
            coefficients = tuple(parameters.detach().cpu().tolist())
            best = FitResult(Model(start.expression, coefficients), value, True)

    def closure():
        solver.zero_grad()
        loss = objective(start, parameters) + (parameters * 0).sum()
        remember(loss)
        loss.backward()
        if not bool(torch.isfinite(parameters.grad).all()):
            raise DomainError('nonfinite coefficient gradient')
        return loss

    try:
        for _ in range(steps):
            solver.step(closure)
            if not bool(torch.isfinite(parameters).all()):
                raise DomainError('nonfinite optimizer iterate')

            # capture final iterates that the optimizer may not have evaluated
            remember(objective(start, parameters))
    except DomainError:
        pass

    return best
