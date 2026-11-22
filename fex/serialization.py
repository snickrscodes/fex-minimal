import json
from pathlib import Path

from .expression import Expression, Model, Node

def save_model(model: Model, path: str | Path) -> None:
    e = model.expression
    info = {
        "nodes": [
            {
                "op": n.op,
                "args": n.args,
                "value": n.value,
                "transform": n.transform,
            }
            for n in e.nodes
        ],
        "outputs": e.outputs,
        "input_dim": e.input_dim,
        "parameter_count": e.parameter_count,
        "parameters": model.parameters,
    }

    data = json.dumps(info, indent=2, allow_nan=False)
    Path(path).write_text(data + "\n", encoding="utf-8")


def load_model(path: str | Path) -> Model:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    nodes = tuple(Node(n["op"], tuple(n["args"]), n["value"], n["transform"]) for n in data["nodes"])
    expression = Expression(nodes, tuple(data["outputs"]), data["input_dim"], data["parameter_count"])
    return Model(expression, tuple(data["parameters"]))