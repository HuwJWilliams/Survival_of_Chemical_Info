"""
Running individual property prediction (PP) analysis
"""

# %% ===== Python Imports =====
import argparse
import sys
from pathlib import Path

import pandas as pd

# %% ===== Project Imports & Pathing Setup=====
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR / "config"))
sys.path.insert(0, str(RUN_DIR / "fns"))

from config import (
    SRC_DIR,
    PATHING_JSON_PATH,
    SUPPORTED_FEATURE_SETS,
    TARGET_COLUMNS,
)
from pp_analysis_fns import *

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

v = Visualise(save_all=False)

FULL_PATHING = getPaths(PATHING_JSON_PATH)
RESULTS_DIR = FULL_PATHING["imp_dirs"]["results_dir"]
PREDS_DIR = FULL_PATHING["prediction_output_dirs"]["rf"]
AVAILABLE_FEATURE_SETS = sorted(set(SUPPORTED_FEATURE_SETS).union(
    feature for property_paths in PREDS_DIR.values() for feature in property_paths
))

# %% ===== Argument Parsing =====

parser = argparse.ArgumentParser(
    description="Generating single property prediction analysis"
)

parser.add_argument(
    "--properties",
    nargs="+",
    default=FULL_PATHING["prediction_output_dirs"]["rf"].keys(),
    choices=FULL_PATHING["prediction_output_dirs"]["rf"].keys(),
    help=f"Name of the experiments ran.",
)

parser.add_argument(
    "--feature-sets",
    nargs="+",
    default=None,
    choices=AVAILABLE_FEATURE_SETS,
    help="Feature sets to analyse; defaults to names configured for the selected properties.",
)

parser.add_argument(
    "--save-dir",
    default=str(RESULTS_DIR / "pp_analysis"),
    help="Where to plot the results",
)

parser.add_argument(
    "--analysis-metrics",
    nargs="+",
    default=["pearson_r"],
    choices=["r2", "pearson_r", "bias", "sdep", "rmse"],
    help="Analysis metrics to plot",
)

args = parser.parse_args()
args.properties, feature_sets = selectPPAnalysisInputs(
    PREDS_DIR, FULL_PATHING["targets"], TARGET_COLUMNS, args.properties, args.feature_sets,
)
if not args.properties:
    parser.error("No selected properties have both a target column and target path configured.")


# %% ===== Loading the Data =====
full_performance_dict = {}
save_path = Path(args.save_dir).expanduser()
save_path.mkdir(parents=True, exist_ok=True)

colour_map = getFeatureColourConfig(feature_ls=feature_sets, colour_map=v.colour_map)


for prop in args.properties:
    try:
        property_pathing = PREDS_DIR[prop]

        int_property_performance_df, ext_property_performance_df = (
            getPropertyPerformanceDfs(
                property_pathing=property_pathing,
                feature_sets=feature_sets,
            )
        )

        full_performance_dict[prop] = {
            "internal": int_property_performance_df,
            "external": ext_property_performance_df,
        }
    except Exception as e:
        print(f"Could not load performance for {prop}:\n{e}")

# %% ===== Running Analysis =====
# Creating a bar plots for each property & performance metric
internal_summary_rows = []
external_summary_rows = []
ft_difference_rows = []
summary_metrics = args.analysis_metrics

# %% ===== Running Analysis =====
# Creating a bar plots for each property & performance metric
for prop, int_ext_perfs in full_performance_dict.items():
    print(f"Processing {prop} performance...")

    for split_name, perf_df in {
        "internal": int_ext_perfs["internal"],
        "external": int_ext_perfs["external"],
    }.items():
        print(f"Processing {split_name}...")
        if perf_df.empty:
            print(f"No performance results for {prop} / {split_name}")
            continue

        summary_metric_cols = {
            metric: getMetricColumn(metric=metric, data=perf_df)
            for metric in summary_metrics
        }
        summary_metric_cols = {
            metric: metric_col
            for metric, metric_col in summary_metric_cols.items()
            if metric_col is not None
        }

        if summary_metric_cols:
            summary_df = perf_df.loc[
                perf_df["stat"] == "mean",
                ["feature_set"] + list(summary_metric_cols.values()),
            ].copy()
            summary_df = summary_df.rename(
                columns={
                    metric_col: metric
                    for metric, metric_col in summary_metric_cols.items()
                    if metric_col != metric
                }
            )

            summary_df["property"] = prop
            summary_df["split"] = split_name

            summary_df = summary_df[
                ["property", "split", "feature_set"] + list(summary_metric_cols.keys())
            ]

            if split_name == "internal":
                internal_summary_rows.append(summary_df)
            elif split_name == "external":
                external_summary_rows.append(summary_df)

        for metric in summary_metrics:

            metric_col = getMetricColumn(metric=metric, data=perf_df)

            if metric_col is None:
                print(f"{metric} not in {split_name} df columns")
                continue

            mean_df = perf_df.loc[
                perf_df["stat"] == "mean",
                ["feature_set", metric_col],
            ].copy()

            std_df = perf_df.loc[
                perf_df["stat"] == "std",
                ["feature_set", metric_col],
            ].copy()

            plot_df = mean_df.merge(
                std_df,
                on="feature_set",
                suffixes=("", "_std"),
            )

            std_metric_col = f"{metric_col}_std"

            plot_df[metric_col] = pd.to_numeric(plot_df[metric_col], errors="coerce")
            plot_df[std_metric_col] = pd.to_numeric(
                plot_df[std_metric_col],
                errors="coerce",
            )

            plot_df = plot_df.dropna(subset=[metric_col, std_metric_col])

            if plot_df.empty:
                print(f"No valid data for {prop} / {split_name} / {metric_col}")
                continue

            ascending = metric_col.lower() in ["rmse", "mse", "mae", "bias", "sdep"]
            plot_df = plot_df.sort_values(metric_col, ascending=ascending)

            if metric_col.lower() == "bias" or metric_col.lower() == "rmse":
                y_lim = None
            else:
                y_lim = (0, 1)

            v.plotBar(
                data=plot_df,
                x_label="feature_set",
                y_label=metric_col,
                y_err_label=std_metric_col,
                title=f"{prop}: {split_name} {metric_col}",
                save_plot=True,
                save_path=save_path / prop / split_name,
                save_fname=f"{prop}_{split_name}_{metric_col}_bar",
                colour_map=colour_map,
                y_lims=y_lim,
            )

# %% ===== True vs Predicted Plots =====
true_vs_pred_rows = []
target_3xiqr_rows = []

for prop in args.properties:
    if prop not in PREDS_DIR:
        print(f"{prop} not in prediction output paths")
        continue

    target_col = TARGET_COLUMNS[prop]
    target_path = FULL_PATHING["targets"][prop]

    try:
        target_df = pd.read_csv(target_path, index_col="ID")
    except Exception as e:
        print(f"Could not load target data for {prop}: {e}")
        continue

    if target_col not in target_df.columns:
        print(f"{target_col} not found in target data for {prop}")
        continue

    target_3xiqr_metrics = plotTarget3xIQRDistribution(
        target_df=target_df,
        target_col=target_col,
        save_path=save_path / prop / "external_3xIQR" / "target_distribution",
        save_fname=f"{prop}_{target_col}_target_distribution_3xIQR",
    )

    if target_3xiqr_metrics:
        target_3xiqr_rows.append(
            {
                "property": prop,
                "target_column": target_col,
                **target_3xiqr_metrics,
            }
        )

    for feature_set in feature_sets:
        if feature_set not in PREDS_DIR[prop]:
            print(f"{feature_set} not in prediction output paths for {prop}")
            continue

        pred_path = Path(PREDS_DIR[prop][feature_set]) / "last_20pct_pred.csv.gz"

        if not pred_path.exists():
            print(f"Missing predictions for {prop} / {feature_set}: {pred_path}")
            continue

        try:
            pred_df = pd.read_csv(pred_path, index_col="ID")
        except Exception as e:
            print(f"Could not load predictions for {prop} / {feature_set}: {e}")
            continue

        pred_col = target_col if target_col in pred_df.columns else pred_df.columns[0]

        metrics = plotTrueVsPred(
            true_df=target_df,
            pred_df=pred_df,
            true_col=target_col,
            pred_col=pred_col,
            save_path=save_path / prop / "external" / "true_vs_pred",
            save_fname=f"{prop}_{feature_set}_true_vs_pred",
            model_name=f"{prop} / {feature_set}",
        )

        row = {
            "property": prop,
            "feature_set": feature_set,
        }

        dist_metrics = {}

        split_ids_path = (
            Path(PREDS_DIR[prop][feature_set])
            / "repeats"
            / "repeat_001"
            / "training_data"
            / "split_train_ids.csv"
        )

        if split_ids_path.exists():
            train_ids = pd.read_csv(split_ids_path)["ID"].astype(str)
            target_for_split = target_df.copy()
            target_for_split.index = target_for_split.index.astype(str)

            train_df = target_for_split.loc[
                target_for_split.index.intersection(train_ids)
            ]

            dist_metrics = plotTrainTestPredDistribution(
                train_df=train_df,
                true_df=target_df,
                pred_df=pred_df,
                train_col=target_col,
                true_col=target_col,
                pred_col=pred_col,
                save_path=save_path / prop / "external" / "prediction_distributions",
                save_fname=f"{prop}_{feature_set}_train_test_pred_distribution",
            )
        else:
            print(
                f"Missing train split IDs for {prop} / {feature_set}: {split_ids_path}"
            )

        if metrics:
            row.update(metrics)

        if dist_metrics:
            row.update(dist_metrics)

        if metrics or dist_metrics:
            true_vs_pred_rows.append(row)

if true_vs_pred_rows:
    pd.DataFrame(true_vs_pred_rows).to_csv(
        save_path / "external_true_vs_pred_metrics.csv",
        index=False,
    )

if target_3xiqr_rows:
    pd.DataFrame(target_3xiqr_rows).to_csv(
        save_path / "target_3xIQR_summary.csv",
        index=False,
    )

if internal_summary_rows:
    internal_summary_df = pd.concat(internal_summary_rows, ignore_index=True)
    internal_summary_df.to_csv(
        save_path / "internal_average_performances.csv",
        index=False,
    )
    plotSummaryHeatmaps(
        summary_df=internal_summary_df,
        split_name="internal",
        save_path=save_path,
        metrics=summary_metrics,
        feature_order=feature_sets,
    )
    plotGroupedPropertyFeatureBar(
        summary_df=internal_summary_df,
        split_name="internal",
        save_path=save_path,
        metric="pearson_r",
        feature_order=feature_sets,
        colour_map=colour_map,
    )

if external_summary_rows:
    external_summary_df = pd.concat(external_summary_rows, ignore_index=True)
    external_summary_df.to_csv(
        save_path / "external_average_performances.csv",
        index=False,
    )
    plotSummaryHeatmaps(
        summary_df=external_summary_df,
        split_name="external",
        save_path=save_path,
        metrics=summary_metrics,
        feature_order=feature_sets,
    )
    plotGroupedPropertyFeatureBar(
        summary_df=external_summary_df,
        split_name="external",
        save_path=save_path,
        metric="pearson_r",
        feature_order=feature_sets,
        colour_map=colour_map,
    )

lipinski_external_df = getLipinskiFilteredExternalPerformanceDf(
    properties=args.properties,
    feature_sets=feature_sets,
    full_pathing=FULL_PATHING,
    preds_dir=PREDS_DIR,
    target_columns=TARGET_COLUMNS,
)

if not lipinski_external_df.empty:
    lipinski_external_df.to_csv(
        save_path / "external_lipinski_average_performances.csv",
        index=False,
    )
    plotGroupedPropertyFeatureBar(
        summary_df=lipinski_external_df,
        split_name="external_lipinski",
        save_path=save_path,
        metric="pearson_r",
        feature_order=feature_sets,
        colour_map=colour_map,
    )


iqr_external_df = get3xIQRFilteredExternalPerformanceDf(
    properties=args.properties,
    feature_sets=feature_sets,
    full_pathing=FULL_PATHING,
    preds_dir=PREDS_DIR,
    target_columns=TARGET_COLUMNS,
)

if not iqr_external_df.empty:
    iqr_external_df.to_csv(
        save_path / "external_3xIQR_average_performances.csv",
        index=False,
    )

    plotSummaryHeatmaps(
        summary_df=iqr_external_df,
        split_name="external_3xIQR",
        save_path=save_path,
        metrics=summary_metrics,
        feature_order=feature_sets,
    )

    for metric in summary_metrics:
        metric_col = getMetricColumn(metric=metric, data=iqr_external_df)

        if metric_col is None:
            print(f"{metric} not in external_3xIQR df columns")
            continue

        for prop in args.properties:
            prop_df = iqr_external_df.loc[iqr_external_df["property"] == prop].copy()

            if prop_df.empty:
                continue

            prop_df[metric_col] = pd.to_numeric(prop_df[metric_col], errors="coerce")
            prop_df = prop_df.dropna(subset=[metric_col])

            if prop_df.empty:
                continue

            ascending = metric_col.lower() in ["rmse", "mse", "mae", "bias", "sdep"]
            prop_df = prop_df.sort_values(metric_col, ascending=ascending)

            if metric_col.lower() in ["bias", "rmse", "sdep"]:
                y_lim = None
            else:
                y_lim = (0, 1)

            v.plotBar(
                data=prop_df,
                x_label="feature_set",
                y_label=metric_col,
                title=f"{prop}: external_3xIQR {metric_col}",
                save_plot=True,
                save_path=save_path / prop / "external_3xIQR",
                save_fname=f"{prop}_external_3xIQR_{metric_col}_bar",
                colour_map=colour_map,
                y_lims=y_lim,
            )

        plotGroupedPropertyFeatureBar(
            summary_df=iqr_external_df,
            split_name="external_3xIQR",
            save_path=save_path,
            metric=metric_col,
            feature_order=feature_sets,
            colour_map=colour_map,
        )

print("Internal FT differences skipped: saved aggregate scores cannot be matched by molecule ID.")
matched_performance_rows = []
for prop in args.properties:
    if not Path(FULL_PATHING["targets"][prop]).exists():
        print(f"Skipping matched FT comparison: missing targets for {prop}")
        continue
    matched_df = getMatchedFTPerformanceDf(
        prop, feature_sets, FULL_PATHING, PREDS_DIR, TARGET_COLUMNS, save_path,
    )
    if matched_df.empty:
        continue
    matched_performance_rows.append(matched_df)
    for split_name, split_df in matched_df.groupby("split"):
        for metric in summary_metrics:
            metric_col = getMetricColumn(metric, split_df)
            if metric_col is None:
                continue
            ft_diff_df = plotFTDifferenceBar(
                data=split_df, prop=prop, split_name=split_name,
                metric_col=metric_col, save_path=save_path, colour_map=colour_map,
            )
            if not ft_diff_df.empty:
                ft_difference_rows.append(ft_diff_df)

if matched_performance_rows:
    pd.concat(matched_performance_rows, ignore_index=True).to_csv(
        save_path / "ft_matched_performances.csv", index=False,
    )

if ft_difference_rows:
    ft_difference_df = pd.concat(ft_difference_rows, ignore_index=True)
    ft_difference_df.to_csv(
        save_path / "ft_differences.csv",
        index=False,
    )
    plotFTDifferenceSummaryBars(
        ft_difference_df=ft_difference_df,
        save_path=save_path,
        metrics=summary_metrics,
        colour_map=colour_map,
    )
else:
    print(
        "No FT comparison summary was generated. Both fine-tuned and pretrained "
        "features must be selected and have last_20pct_pred.csv.gz files with "
        "at least two shared valid molecule IDs and a defined comparison metric. "
        "See the missing-prediction and shared-molecule messages above."
    )
