"""Small integration tests for APIs used by external callers."""

from pathlib import Path
import tempfile
import unittest

import torch

from fex import (
    DAGBuilder, DomainError, FitStage, Model, Regression,
    evaluate, fit_stages, format_model, load_model, save_model,
)


class APITests(unittest.TestCase):
    def test_select_outputs_does_not_evaluate_other_regions(self):
        b = DAGBuilder(1, affine_unary=False)
        x = b.input(0)
        invalid = b.unary("sqrt", b.unary("neg", x))
        b.output(invalid, x, x)

        model = Model(b.build(), ())
        points = torch.tensor(
            [[1.0], [4.0]], dtype=torch.float64
        )

        with self.assertRaises(DomainError):
            evaluate(model, points)

        selected = model.select_outputs((1, 2))
        torch.testing.assert_close(
            evaluate(selected, points),
            points.repeat(1, 2),
        )
        self.assertEqual(
            model.expression.outputs, (2, 0, 0)
        )

    def test_staged_fit(self):
        b = DAGBuilder(1)
        b.output(b.unary("square", b.input(0)))

        x = torch.linspace(
            -1, 1, 11, dtype=torch.float64
        )[:, None]
        objective = Regression(
            x, 1.7 * x.square() + 0.3
        )

        stages = (
            FitStage(
                lambda p: torch.optim.Adam(
                    [p], lr=0.1
                ),
                20,
            ),
            FitStage(
                lambda p: torch.optim.LBFGS(
                    [p],
                    max_iter=10,
                    max_eval=15,
                    line_search_fn="strong_wolfe",
                ),
                10,
            ),
        )

        result = fit_stages(
            Model(b.build(), (0.0, 0.0)),
            objective,
            stages,
        )
        self.assertTrue(result.valid)
        self.assertLess(result.loss, 1e-6)

    def test_format_and_json_round_trip(self):
        b = DAGBuilder(2, affine_unary=False)
        leaf = b.vsum("square")
        b.output(b.unary("sqrt", leaf), leaf)

        model = Model(
            b.build(), (0.1, 0.2, 2.0)
        )

        rendered = format_model(
            model, variables=("x", "y")
        )
        self.assertEqual(len(rendered), 2)
        self.assertEqual(
            rendered[0], f"sqrt({rendered[1]})"
        )
        self.assertIn("square(x)", rendered[1])

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            save_model(model, path)
            restored = load_model(path)

        self.assertEqual(restored, model)

        x = torch.tensor(
            [[0.5, 0.2], [0.0, -0.3]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(
            evaluate(model, x),
            evaluate(restored, x),
        )


if __name__ == "__main__":
    unittest.main()
