from dataclasses import dataclass

import torch
from torch import Tensor

from .leggauss import leggauss_glr


QUAD_NODES = 16
_z, _w = leggauss_glr(QUAD_NODES)
# transform quad rule from [-1, 1] -> [0, 1]
QUAD_POSITIONS = 0.5 * (torch.as_tensor(_z, dtype=torch.float64) + 1.0)
QUAD_WEIGHTS = 0.5 * torch.as_tensor(_w, dtype=torch.float64)


@dataclass(frozen=True)
class Node:
    # "input", "param", "literal", or an explicit operator ("D" or "I" okay too)
    op: str

    # arguments (indices of nodes in the graph) for the op
    # refers backward, for "D" and "I" nodes a single arg for the variable (being differentiated or integrated)
    args: tuple[int, ...] = ()

    # either input coordinate index (for "input"), param index (for "param")
    # a constant (for "literal"), or (for "D" and "I") coordinate on which the op acts, for other operators leave None
    value: float | int | None = None


@dataclass(frozen=True)
class Expression:
    # represented as a topologically sorted DAG, evaluation uses forward traversal
    # nodes refer backward, outputs are node indices
    nodes: tuple[Node, ...]
    outputs: tuple[int, ...]
    input_dim: int
    parameter_count: int = 0


@dataclass(frozen=True)
class Model:
    # holds an expression and parameter values
    expression: Expression
    parameters: tuple[float, ...]


class DomainError(ValueError):
    """a mathematically legal candidate is undefined or nonfinite on some rows"""


def _protected_divide(numerator, denominator):
    # clamps 0 < denominator < 1e-6 to 1e-6, and -1e-6 < denominator < 0 to -1e-6
    floor = torch.ones_like(denominator) * 1e-6
    safe = torch.where(
        denominator.abs() >= 1e-6,
        denominator, torch.where(denominator < 0, -floor, floor),
    )
    return numerator / safe


# every function in OPERATIONS must be pointwise, must not modify its operands, and must support torch autodiff
# the caller configures ops before eval and does not modify the table during an eval
# D/I nodes use value = coordinate, and re-evaluate their scalar child
# I integrates from 0 to coordinate with GL quad
# gradients differentiate the quadrature rule (not integral itself)
# strict div/log and pdiv/plog (protected) are different
OPERATIONS = {
    'add': torch.add,
    'sub': torch.sub,
    'mul': torch.mul,
    'div': torch.div,
    'pdiv': _protected_divide,
    'neg': torch.neg,
    'square': torch.square,
    'sin': torch.sin,
    'cos': torch.cos,
    'exp': torch.exp,
    'tanh': torch.tanh,
    'log': torch.log,
    'plog': lambda a: torch.log(torch.maximum(a, torch.ones_like(a) * 1e-6)),
    'ge': torch.ge,
}


def evaluate(model: Model, inputs: Tensor, *, parameters: Tensor | None = None):
    # inputs: fp32 (N, D), D is number of coordinates
    # parameters should be on the same device and dtype as inputs
    expression = model.expression
    p = parameters if parameters is not None else torch.tensor(model.parameters, dtype=inputs.dtype, device=inputs.device)
    return _evaluate(expression, expression.outputs, p, inputs)


def _evaluate(expression, roots, p, x):
    VISIT, APPLY, BRANCH, MERGE = range(4)
    # a scope is (indices into this call's x, cached node values)
    root_scope = (torch.arange(len(x), device=x.device), {})

    for output in roots:
        # work is a stack of tasks to do (VISIT, APPLY, BRANCH, MERGE)
        # each task is (type, node index, scope, saved information)
        work = [(VISIT, output, root_scope, None)]  # last argument is saved info, only applies to BRANCH and MERGE tasks
        while work:
            action, index, scope, saved = work.pop()
            rows, cache = scope
            if index in cache:
                continue
            node = expression.nodes[index]

            if action == VISIT:
                if node.op == 'input':
                    value = x[:, node.value] if scope is root_scope else x[rows, node.value]
                elif node.op in ('param', 'literal'):
                    scalar = p[node.value] if node.op == 'param' else node.value
                    base = x[:, 0] if scope is root_scope else x[rows, 0]
                    value = torch.ones_like(base) * scalar
                elif node.op in ('D', 'I'):
                    if torch.is_inference_mode_enabled():
                        raise NotImplementedError('D/I nodes do not support inference_mode, use no_grad')
                    coordinates = x if scope is root_scope else x[rows]
                    value = _function_value(expression, node, p, coordinates)
                elif node.op == 'where':
                    # evaluate the condition first and get the mask
                    # use the mask to create separate true and false row scopes
                    # evaluate the subgraphs on those rows, and then merge the results
                    work.append((BRANCH, index, scope, None))
                    work.append((VISIT, node.args[0], scope, None))
                    continue
                else:
                    work.append((APPLY, index, scope, None))
                    for operand in node.args:
                        work.append((VISIT, operand, scope, None))
                    continue

            elif action == APPLY:
                args = [cache[operand] for operand in node.args]
                value = OPERATIONS[node.op](*args)

            elif action == BRANCH:
                mask = cache[node.args[0]]
                inverse_mask = ~mask
                # two scopes
                branches = (
                    (rows[mask], {}),
                    (rows[inverse_mask], {}),
                )
                work.append((MERGE, index, scope, (mask, branches)))
                # push true before false to preserve LIFO
                for branch, operand in zip(branches, node.args[1:]):
                    if len(branch[0]):
                        work.append((VISIT, operand, branch, None))
                continue

            else:  # MERGE
                mask, branches = saved
                base = x[:, 0] if scope is root_scope else x[rows, 0]
                value = torch.zeros_like(base)
                # loop is just for the (2) true and false branches, node has 3 args total
                for selection, operand, branch in zip((mask, ~mask), node.args[1:], branches):
                    branch_rows, branch_cache = branch
                    if len(branch_rows):
                        value[selection] = branch_cache[operand]  # branch evaluated on its active rows
                    # no future task needs this completed branch's cache
                    branch_cache.clear()

            if not bool(torch.isfinite(value).all()):
                raise DomainError(f'nonfinite values at {node.op}')
            # cache the value if needed in future (such as for other outputs)
            cache[index] = value

    return torch.stack([root_scope[1][index] for index in roots], dim=1)


# exclusively for evaluating D and I nodes
def _function_value(expression, node, parameters, coordinates):
    axis = node.value
    child = node.args  # single child (tuple with 1 element)
    if node.op == 'D':
        outer_grad = torch.is_grad_enabled()
        with torch.enable_grad():
            # don't detach anything, preserve graph for future chain rules on outer queries
            # only differentiate wrt the query, not whatever produced it
            query = coordinates.clone().requires_grad_(True)
            values = _evaluate(expression, child, parameters, query)[:, 0]
            gradient = None
            if values.requires_grad:
                # values.sum() works because input rows are independent (only depends on its own row)
                gradient = torch.autograd.grad(values.sum(), query, create_graph=True, allow_unused=True)[0]

            # evil trick: if the gradient is 0, keep it connected to the graph for future (higher-order) derivatives
            zero = query[:, axis] * 0 + (parameters * 0).sum()
            result = zero if gradient is None else gradient[:, axis] + zero

        # no_grad callers need the derivative's value, not its computational graph
        # nested D nodes execute inside the enabled region and retain their graphs
        return result if outer_grad else result.detach()

    # I node, apply GL quadrature
    # construct query tensor (to evaluate at GL nodes, only integration axis varies - other free coordinates are fixed)
    # apply weighted sum - don't special case endpoint cases (endpoint is 0) - destroys graph info
    # future derivatives are on the quad rule itself
    s = QUAD_POSITIONS.to(coordinates)
    w = QUAD_WEIGHTS.to(coordinates)
    query = coordinates[:, None, :].expand(-1, QUAD_NODES, -1).clone()
    query[:, :, axis] = coordinates[:, axis, None] * s
    values = _evaluate(expression, child, parameters, query.reshape(-1, coordinates.shape[1]))[:, 0]
    return coordinates[:, axis] * (values.reshape(len(coordinates), QUAD_NODES) * w).sum(dim=1)


def derivatives(model: Model, inputs: Tensor, *, order: int = 1, parameters: Tensor | None = None):
    # returns value, with some derivatives
    # value: (N, O), O is number of outputs
    # jacobian: (N, O, D), partials of each output fn wrt each of its coordinates, D is number of coordinates
    # hessian (only if order = 2), (N, O, D, D), at index ij is mixed partial ij (in lagrange notation)

    # leaves parameter differentiation possible (same trick as for D nodes), but only use this with gradients enabled
    x = inputs if inputs.requires_grad else inputs.detach().clone().requires_grad_(True)
    p = parameters if parameters is not None else torch.tensor(model.parameters, dtype=inputs.dtype, device=inputs.device)
    # zero links keep constant/unused-coordinate derivatives connected to
    # inputs and coefficients, including further derivatives of exact zeros
    anchor = (p * 0).sum()  # multiply before sum (prevent overflow)
    values = evaluate(model, x, parameters=p) + anchor

    def gradient(value):
        result = None
        if value.requires_grad:
            # same value.sum() trick as for D nodes
            result = torch.autograd.grad(value.sum(), x, create_graph=True, retain_graph=True, allow_unused=True)[0]
        # apply same evil trick as for D nodes, leave the 0 (or None) gradient connected to the graph for future (upstream) grads
        if result is None:  # no dependency in graph
            # gradient is technically 0, leave it connected to graph
            return x * 0 + anchor
        return result + x * 0 + anchor

    jacobian = torch.stack([gradient(values[:, output]) for output in range(values.shape[1])], dim=1)
    if order == 1:
        return values, jacobian

    hessian = torch.stack([
        torch.stack(
            [gradient(jacobian[:, output, coordinate]) for coordinate in range(x.shape[1])], dim=1,
        )
        for output in range(values.shape[1])
    ], dim=1)
    return values, jacobian, hessian
