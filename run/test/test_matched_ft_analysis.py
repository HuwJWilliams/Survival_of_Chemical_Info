"""Exercise matched scoring without importing plotting or local path configuration."""
import ast
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import pandas as pd
except ImportError:
    pd = None


@unittest.skipIf(pd is None, "Requires pandas and numpy")
class MatchedFTAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        source = Path(__file__).resolve().parents[1] / "fns" / "pp_analysis_fns.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"getMatchedFTPerformanceDf", "getFTDifferenceDf", "calculateR2"}
        # These pure data helpers are isolated from module-level plotting setup.
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        nodes += [node for node in tree.body if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "FT_FEATURE_PAIRS"
                          for target in node.targets)]
        self.ns = {"pd": pd, "np": np, "Path": Path}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), self.ns)
        self.base = "chemberta-dc"
        self.ft = "ft-scaffold-chemberta-dc"
        self.paths = {"bp": {}}
        self.features = [self.base, self.ft]
        self.target = self.root / "targets.csv"
        pd.DataFrame({"ID": ["001", "002", "003", "004", "005"],
                      "value": [1., 2., 3., 4., 100.]}).to_csv(self.target, index=False)

    def predictions(self, feature, ids, values):
        directory = self.root / feature
        directory.mkdir(exist_ok=True)
        self.paths["bp"][feature] = directory
        pd.DataFrame({"ID": ids, "value": values}).to_csv(
            directory / "last_20pct_pred.csv.gz", index=False)

    def score(self):
        return self.ns["getMatchedFTPerformanceDf"](
            "bp", self.features, {"targets": {"bp": self.target}},
            self.paths, {"bp": "value"}, self.root / "output")

    def test_extra_molecules_do_not_change_pair_scores(self):
        self.predictions(self.base, ["001", "002", "003", "004", "005"], [1, 2, 3, -500, 500])
        self.predictions(self.ft, ["003", "001", "002"], [3, 1, 2])
        result = self.score()
        self.assertTrue((result["n"] == 3).all())
        self.assertTrue(np.allclose(result["pearson_r"], 1))
        diff = self.ns["getFTDifferenceDf"](
            result.loc[result["split"] == "external"], "bp", "external", "pearson_r")
        self.assertEqual(diff.iloc[0]["difference"], 0)
        self.assertEqual(diff.iloc[0]["n_matched"], 3)
        audit = next((self.root / "output/bp/external/ft_differences").glob("*.csv"))
        self.assertEqual(set(pd.read_csv(audit, dtype={"ID": str})["ID"]), {"001", "002", "003"})

    def test_invalid_predictions_removed_from_both_and_iqr_applied(self):
        self.predictions(self.base, ["001", "002", "003", "004", "005"], [1, 2, np.inf, 4, 100])
        self.predictions(self.ft, ["001", "002", "003", "004", "005"], [1, 2, 3, np.nan, 100])
        result = self.score()
        self.assertEqual(result.groupby("split")["n"].first().to_dict(),
                         {"external": 3, "external_3xIQR": 2})

    def test_each_pair_uses_its_own_intersection(self):
        other = "ft-random-chemberta-dc"
        self.features.append(other)
        self.predictions(self.base, ["001", "002", "003", "004"], [1, 2, 4, 3])
        self.predictions(self.ft, ["001", "002"], [1, 2])
        self.predictions(other, ["003", "004"], [3, 4])
        result = self.score()
        base = result.loc[(result["split"] == "external") & (result["feature_set"] == self.base)]
        scores = base.set_index("comparison_pair")["pearson_r"]
        self.assertAlmostEqual(scores[self.ft], 1)
        self.assertAlmostEqual(scores[other], -1)

    def test_missing_or_insufficient_predictions_skip_pair(self):
        self.predictions(self.base, ["001", "002"], [1, 2])
        self.assertTrue(self.score().empty)
        self.predictions(self.ft, ["002", "003"], [2, 3])
        self.assertTrue(self.score().empty)

    def test_duplicate_ids_rejected(self):
        self.predictions(self.base, ["001", "001"], [1, 2])
        self.predictions(self.ft, ["001", "002"], [1, 2])
        with self.assertRaisesRegex(ValueError, "Duplicate prediction"):
            self.score()

    def test_aggregate_scores_rejected(self):
        with self.assertRaisesRegex(ValueError, "matched molecule IDs"):
            self.ns["getFTDifferenceDf"](
                pd.DataFrame({"stat": ["mean"], "feature_set": [self.base], "pearson_r": [0.9]}),
                "bp", "external", "pearson_r")


if __name__ == "__main__":
    unittest.main()
