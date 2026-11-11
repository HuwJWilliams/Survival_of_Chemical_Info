"""Checks for the statistics reported across held-out resamples."""

import math
import sys
from pathlib import Path

from sklearn.model_selection import GridSearchCV, KFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "src" / "models"))
from model_base import ClassifierBase, RegressorBase


def _model(model_class):
    return model_class(
        cv_function=KFold,
        hp_search_function=GridSearchCV,
        cv_kwargs={"n_splits": 5},
        hp_search_kwargs={"refit": True},
    )


def test_regression_reports_sample_std_and_range():
    metrics = ("bias", "sdep", "mse", "rmse", "r2", "r_pearson", "r_spearman")
    results = [{metric: score for metric in metrics} for score in (1.0, 2.0, 3.0)]

    summary = _model(RegressorBase)._aggregate_performance(results)

    assert summary["RMSE"] == 2.0
    assert summary["RMSE_std"] == 1.0
    assert summary["RMSE_min"] == 1.0
    assert summary["RMSE_max"] == 3.0
    assert summary["RMSE_range"] == 2.0


def test_single_split_and_missing_classification_score():
    regression_metrics = (
        "bias", "sdep", "mse", "rmse", "r2", "r_pearson", "r_spearman"
    )
    one_split = [{metric: 2.0 for metric in regression_metrics}]
    summary = _model(RegressorBase)._aggregate_performance(one_split)

    assert math.isnan(summary["RMSE_std"])
    assert summary["RMSE_range"] == 0.0

    classifier = _model(ClassifierBase)
    classifier._n_classes = 2
    classification_metrics = (
        "acc", "bal_acc", "sens", "spec", "ppv", "npv", "auc", "mcc"
    )
    results = [{metric: 0.5 for metric in classification_metrics} for _ in range(3)]
    for result, score in zip(results, (0.8, float("nan"), 1.0)):
        result["auc"] = score
        result["npv"] = float("nan")

    summary = classifier._aggregate_performance(results)

    assert summary["AUC"] == 0.9
    assert summary["AUC_std"] == 0.1414
    assert summary["AUC_min"] == 0.8
    assert summary["AUC_max"] == 1.0
    assert summary["AUC_range"] == 0.2
    assert math.isnan(summary["NPV"])
    assert math.isnan(summary["NPV_std"])
    assert math.isnan(summary["NPV_range"])
