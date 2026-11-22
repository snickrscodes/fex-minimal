from .expression import Model

_INFIX = {'add': '+', 'sub': '-', 'mul': '*', "pow": "**", 'div': '/', 'ge': '>='}

def format_model(model: Model, *, variables: tuple[str, ...] | None = None, digits: int = 10) -> tuple[str, ...]:
    # returns one human-readable mathematical string per output
    expression = model.expression
    if variables is None:
        variables = tuple(f'x{i}' for i in range(expression.input_dim))

    def number(value):
        return format(value, f'.{digits}g')

    cache: dict[int, str] = {}

    def render(i: int) -> str:
        if i in cache:
            return cache[i]
        node = expression.nodes[i]
        op = node.op
        if op == 'input':
            result = variables[node.value]
        elif op == 'param':
            result = number(model.parameters[node.value])
        elif op == 'literal':
            result = number(node.value)
        elif op == 'vsum':
            start = node.value
            terms = []
            for coordinate, name in enumerate(variables):
                feature = name if node.transform is None else f'{node.transform}({name})'
                terms.append(f'({number(model.parameters[start + coordinate])}*{feature})')
            terms.append(f'({number(model.parameters[start + expression.input_dim])})')
            result = '(' + ' + '.join(terms) + ')'
        elif op in ('D', 'I'):
            axis = variables[node.value]
            result = f'D[{axis}]' if op == 'D' else f'I[0,{axis}]'
            result = result + '(' + render(node.args[0]) + ')'
        elif op in _INFIX:
            lhs, rhs = node.args
            result = f'({render(lhs)} {_INFIX[op]} {render(rhs)})'
        else:
            result = op + '(' + ', '.join(render(a) for a in node.args) + ')'
        cache[i] = result
        return result

    return tuple(render(i) for i in expression.outputs)
