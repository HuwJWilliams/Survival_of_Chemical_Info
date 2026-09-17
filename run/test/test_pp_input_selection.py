"""Input/configuration regressions; no plotting or local paths required."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


class PPInputSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).resolve().parents[1] / "fns" / "pp_analysis_fns.py"
        tree = ast.parse(source.read_text(encoding="utf-8"))
        names = {"selectPPAnalysisInputs", "getPropertyPerformanceDfs"}
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        # Capture the loader's records directly; it performs no DataFrame operations.
        cls.ns = {"Path": Path, "json": json, "pd": SimpleNamespace(DataFrame=list)}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), "exec"), cls.ns)

    def test_legacy_names_discovered_and_unconfigured_property_skipped(self):
        properties, features = self.ns["selectPPAnalysisInputs"](
            {"bp": {"chemberta": "a", "ft-scaffold-chemberta": "b"},
             "ld50": {"rdkit": "c"}},
            {"bp": "target.csv", "ld50": "legacy.csv"}, {"bp": "Boiling_Point"},
            ["bp", "ld50"],
        )
        self.assertEqual(properties, ["bp"])
        self.assertEqual(features, ["chemberta", "ft-scaffold-chemberta"])

    def test_selected_properties_control_feature_discovery(self):
        properties, features = self.ns["selectPPAnalysisInputs"](
            {"bp": {"chemberta": "a"}, "logd": {"molformer-ibm": "b"}},
            {"bp": "a", "logd": "b"}, {"bp": "x", "logd": "y"}, ["logd"],
        )
        self.assertEqual(features, ["molformer-ibm"])

    def test_explicit_selection_preserved_and_random_ft_excluded(self):
        _, features = self.ns["selectPPAnalysisInputs"](
            {"bp": {}}, {"bp": "a"}, {"bp": "x"}, ["bp"],
            ["chemberta", "ft-scaffold-chemberta", "ft-random-chemberta"],
        )
        self.assertEqual(features, ["chemberta", "ft-scaffold-chemberta"])

    def test_missing_target_path_skipped(self):
        properties, features = self.ns["selectPPAnalysisInputs"](
            {"bp": {"rdkit": "a"}}, {}, {"bp": "x"}, ["bp"],
        )
        self.assertEqual((properties, features), ([], []))

    def test_missing_and_malformed_results_do_not_discard_valid_features(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            valid, broken, missing = [root / name for name in ("valid", "broken", "missing")]
            valid.mkdir()
            broken.mkdir()
            (valid / "rf_performance.json").write_text(json.dumps({
                split: {"mean": {"pearson_r": 0.8}, "std": {"pearson_r": 0.1}}
                for split in ("internal", "external")
            }))
            (broken / "rf_performance.json").write_text("{invalid")
            internal, external = self.ns["getPropertyPerformanceDfs"](
                {"chemberta": valid, "mordred": broken, "rdkit": missing},
                ["chemberta-dc", "rdkit", "mordred", "chemberta"],
            )
            self.assertEqual(len(internal), 2)
            self.assertEqual(len(external), 2)
            self.assertEqual(internal[0]["feature_set"], "chemberta")
            self.assertEqual(external[0]["pearson_r"], 0.8)


if __name__ == "__main__":
    unittest.main()
