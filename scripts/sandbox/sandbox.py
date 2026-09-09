import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# data = pd.read_csv(
#     "/users/yhb18174/TL_project/datasets/embeddings/BP_embeddings/"
#     "bp_ft-random-molformer-c3-1b_fine_tuned_model/fine_tuning_log_history.csv"
# )

# train_loss = data.dropna(subset=["loss"]).copy()

# x_col = "epoch" if "epoch" in train_loss.columns else "step"

# plt.figure(figsize=(7, 4))
# plt.plot(train_loss[x_col], train_loss["loss"], marker="o")
# plt.xlabel(x_col.capitalize())
# plt.ylabel("Training loss")
# plt.title("Fine-tuning Training Loss")
# plt.tight_layout()
# plt.show()

# %% Checking averages

data = pd.read_csv(
    "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_avg_transformers/descriptor_r2_by_model_with_average.csv",
    index_col=0,
)


def averageExperimentPerformanceTotalDescriptors(
    exp: str,
    exp_dir: str | Path,
    save: bool = True,
) -> pd.DataFrame:
    """
    Average one CFP experiment across all descriptor rows.

    This is similar to averageExperimentPerformance(), but instead of returning
    one averaged row per descriptor, it returns one total averaged row for the
    whole experiment.
    """

    exp_dir = Path(exp_dir)
    exp_path = exp_dir / f"{exp}.csv"

    if not exp_path.exists():
        raise FileNotFoundError(f"Could not find experiment file: {exp_path}")

    exp_df = pd.read_csv(exp_path, index_col=0)

    task_metric_cols = {
        "regression": [
            "Bias",
            "SDEP",
            "MSE",
            "RMSE",
            "r2",
            "Pearson_r",
            "Pearson_p",
        ],
        "binary_classification": [
            "Accuracy",
            "Balanced_Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "AUC",
            "MCC",
        ],
        "multiclass_classification": [
            "Accuracy",
            "Balanced_Accuracy",
            "F1_macro",
            "AUC_OVR",
            "MCC",
        ],
    }

    descriptor_counts = {}
    mean_metrics = {}

    if "task_type" in exp_df.columns:
        for task_name, metric_cols in task_metric_cols.items():
            task_df = exp_df.loc[exp_df["task_type"] == task_name].copy()
            descriptor_counts[f"n_{task_name}_descriptors"] = len(task_df)

            for metric_col in metric_cols:
                if metric_col not in task_df.columns:
                    continue

                metric_values = pd.to_numeric(task_df[metric_col], errors="coerce")
                if metric_values.notna().any():
                    mean_metrics[f"{task_name}_{metric_col}"] = metric_values.mean()
    else:
        numeric_cols = exp_df.select_dtypes(include="number").columns.tolist()
        mean_metrics = exp_df[numeric_cols].mean(numeric_only=True).to_dict()

    exp_parts = exp.split("_")
    total_average_df = pd.DataFrame(
        [
            {
                "experiment": exp,
                "pred": exp_parts[1] if len(exp_parts) > 1 else pd.NA,
                "train": exp_parts[3] if len(exp_parts) > 3 else pd.NA,
                "n_descriptors": len(exp_df),
                **descriptor_counts,
                **mean_metrics,
            }
        ]
    )

    if save:
        total_average_df.to_csv(exp_dir / "total_descriptor_average.csv", index=False)

    return total_average_df


import sys
from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parents[2] / "run" / "config"
sys.path.insert(0, str(CONFIG_DIR))
from config import FULL_PATHING, SRC_DIR

total_average_dfs = []

cfp_block = FULL_PATHING["prediction_output_dirs"][
    "lipinski_cross_feature_predictions"
]["all"]

# for exp, exp_dir in cfp_block.items():
#     exp_parts = exp.split("_")

#     if len(exp_parts) < 4:
#         continue

#     pred_feature = exp_parts[1]

#     if pred_feature != "mordred" and exp != "pred_rdkit_tr_mordred":
#         continue

#     try:
#         total_average_df = averageExperimentPerformanceTotalDescriptors(
#             exp=exp,
#             exp_dir=exp_dir,
#         )
#         total_average_dfs.append(total_average_df)
#     except Exception as e:
#         print(e)

# all_total_average_df = pd.concat(total_average_dfs, ignore_index=True)
# numeric_cols = all_total_average_df.select_dtypes(include="number").columns
# all_total_average_df[numeric_cols] = all_total_average_df[numeric_cols].round(3)
# all_total_average_df.to_csv(
#     "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_all_task_metric_avg_perf.csv",
#     index=False,
# )


# %% Plot top RDKit features for each predicted Mordred descriptor
# %% Plot top RDKit features for predicting Mordred Autocorrelation descriptors
sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import *

# Get all Mordred descriptors belonging to Autocorrelation
res = getMordredGroups()
autocorrelation_descs = res["Autocorrelation"]

v = Visualise(save_all=False)

# original_data = pd.read_csv(
#     cfp_block["pred_mordred_tr_rdkit"] / "all_feature_importance.csv"
# )

# # Corresponding importance columns
# cols_to_avg = [
#     f"Importance_{desc}"
#     for desc in autocorrelation_descs
#     if f"Importance_{desc}" in original_data.columns
# ]

# # Mean importance of each RDKit feature across all
# # Autocorrelation Mordred descriptors
# original_data["Importance_Autocorrelation"] = original_data[cols_to_avg].mean(axis=1)

# v.plotFeatureImportance(
#     data=original_data,
#     x_col="Importance_Autocorrelation",
#     y_col="Feature",
#     top_n=25,
#     save_path=str(Path(__file__).parent),
# )

# %% Plot Mordred atomic-mass autocorrelation descriptor distributions
import math
import re

import numpy as np

FILE_DIR = Path(__file__).resolve().parent
PROJ_DIR = FILE_DIR.parents[1]
AUTOCORR_FAMILIES = ["ATS", "AATS", "ATSC", "AATSC", "MATS", "GATS"]
PROPERTY_SUFFIX = "m"
TRAINING_PERCENTILE = 0.99

# mordred_features_path = Path(FULL_PATHING["full_features"]["all"]["mordred"])
# if not mordred_features_path.exists():
#     mordred_features_path = (
#         PROJ_DIR
#         / "run"
#         / "test"
#         / "expected_test_results"
#         / "expected_mordred_features.csv"
#     )

# mordred_features = pd.read_csv(mordred_features_path, index_col=0, low_memory=False)

# print(f"Loaded Mordred features from: {mordred_features_path}")

# for family in AUTOCORR_FAMILIES:
#     mass_pattern = re.compile(rf"^{family}(?P<lag>\d+){PROPERTY_SUFFIX}(?:_mordred)?$")

#     mass_cols = []
#     for col in mordred_features.columns:
#         match = mass_pattern.match(col)
#         if match:
#             mass_cols.append((int(match.group("lag")), col))

#     mass_cols = [col for _, col in sorted(mass_cols)]

#     if not mass_cols:
#         print(f"Skipping {family}: no atomic-mass descriptors found.")
#         continue

#     mass_values = []
#     percentile_cutoffs = []
#     n_raw_values = []

#     for col in mass_cols:
#         non_nan_values = pd.to_numeric(mordred_features[col], errors="coerce").dropna()
#         percentile_cutoff = non_nan_values.quantile(TRAINING_PERCENTILE)
#         trained_values = non_nan_values.loc[non_nan_values <= percentile_cutoff]

#         mass_values.append(trained_values)
#         percentile_cutoffs.append(percentile_cutoff)
#         n_raw_values.append(len(non_nan_values))

#     all_mass_values = pd.concat(mass_values, ignore_index=True)

#     if all_mass_values.empty:
#         print(
#             f"Skipping {family}: descriptors found, but no values remained after "
#             f"the {TRAINING_PERCENTILE:.0%} cutoff."
#         )
#         continue

#     bin_min = all_mass_values.min()
#     bin_max = all_mass_values.max()
#     if bin_min == bin_max:
#         bin_min -= 0.5
#         bin_max += 0.5

#     bin_edges = np.linspace(bin_min, bin_max, 41)
#     n_plot_cols = 3
#     n_plot_rows = math.ceil(len(mass_cols) / n_plot_cols)

#     fig, axes = plt.subplots(
#         n_plot_rows,
#         n_plot_cols,
#         figsize=(4.8 * n_plot_cols, 3.6 * n_plot_rows),
#         squeeze=False,
#     )
#     axes_flat = axes.ravel()

#     for ax, col, trained_values, percentile_cutoff, n_raw in zip(
#         axes_flat,
#         mass_cols,
#         mass_values,
#         percentile_cutoffs,
#         n_raw_values,
#     ):
#         pct_zero = 100 * (trained_values == 0).mean()
#         pct_removed = 100 * (1 - (len(trained_values) / n_raw)) if n_raw else 0
#         ax.hist(
#             trained_values,
#             bins=bin_edges,
#             edgecolor="black",
#             color="steelblue",
#             alpha=0.85,
#         )
#         ax.set_title(
#             f"{col}\ntrain n={len(trained_values)}, 0={pct_zero:.1f}%, "
#             f"cut>{TRAINING_PERCENTILE:.0%} ({pct_removed:.1f}%)",
#             fontsize=11,
#         )
#         ax.set_xlabel(f"Descriptor value <= p99 ({percentile_cutoff:.3g})")
#         ax.set_ylabel("Count")
#         ax.grid(axis="y", linestyle="--", alpha=0.25)

#     for ax in axes_flat[len(mass_cols) :]:
#         ax.axis("off")

#     fig.suptitle(
#         f"Mordred Autocorrelation: {family} Atomic Mass",
#         fontsize=14,
#         weight="bold",
#     )
#     fig.tight_layout()

#     save_path = FILE_DIR / f"{family.lower()}_mass_histograms.png"
#     fig.savefig(save_path, dpi=200, bbox_inches="tight")
#     plt.close(fig)

#     print(f"Plotted {family} descriptors: {', '.join(mass_cols)}")
#     print(f"Saved histogram plot to: {save_path}")


# %% Compare fine-tuned predictions against ID-trimmed base predictions
#
# This checks whether fine-tuning genuinely changes performance, or whether the
# apparent gain mostly comes from removing the fine-tuning molecules from the
# prediction/evaluation set.

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

CONFIG_DIR = Path(__file__).resolve().parents[2] / "run" / "config"
sys.path.insert(0, str(CONFIG_DIR))
from config import FULL_PATHING, TARGET_COLUMNS


FT_BASE_COMPARISONS = [
    ("molformer-c3-1b", "ft-random-molformer-c3-1b"),
    ("molformer-c3-1b", "ft-scaffold-molformer-c3-1b"),
    ("chemberta", "ft-random-chemberta"),
    ("chemberta", "ft-scaffold-chemberta"),
    ("chembertasey", "ft-random-chembertasey"),
    ("chembertasey", "ft-scaffold-chembertasey"),
    ("selformer", "ft-random-selformer"),
    ("selformer", "ft-scaffold-selformer"),
    ("molformer-dc", "ft-random-molformer-dc"),
    ("molformer-dc", "ft-scaffold-molformer-dc"),
    ("molformer-ibm", "ft-random-molformer-ibm"),
    ("molformer-ibm", "ft-scaffold-molformer-ibm"),
    ("chemberta-dc", "ft-random-chemberta-dc"),
    ("chemberta-dc", "ft-scaffold-chemberta-dc"),
    ("chemberta-sey", "ft-random-chemberta-sey"),
    ("chemberta-sey", "ft-scaffold-chemberta-sey"),
]


def _read_target_and_prediction(
    prop: str,
    feature_set: str,
    full_pathing: dict,
    preds_dir: dict,
    target_columns: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, str, str] | None:
    if prop not in preds_dir:
        print(f"Skipping {prop}: no prediction paths found.")
        return None

    if feature_set not in preds_dir[prop]:
        print(f"Skipping {prop} / {feature_set}: no prediction path found.")
        return None

    if prop not in target_columns:
        print(f"Skipping {prop}: no target column configured.")
        return None

    target_col = target_columns[prop]
    target_path = Path(full_pathing["targets"][prop])
    pred_path = Path(preds_dir[prop][feature_set]) / "last_20pct_pred.csv.gz"

    if not target_path.exists():
        print(f"Skipping {prop}: missing target file: {target_path}")
        return None

    if not pred_path.exists():
        print(f"Skipping {prop} / {feature_set}: missing predictions: {pred_path}")
        return None

    target_df = pd.read_csv(target_path, index_col="ID")
    pred_df = pd.read_csv(pred_path, index_col=0)

    target_df.index = target_df.index.astype(str)
    pred_df.index = pred_df.index.astype(str)

    pred_col = target_col if target_col in pred_df.columns else pred_df.columns[0]

    if target_col not in target_df.columns:
        print(f"Skipping {prop}: {target_col} not found in target file.")
        return None

    return target_df, pred_df, target_col, pred_col


def _make_eval_df(
    target_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    target_col: str,
    pred_col: str,
    keep_ids: pd.Index | None = None,
) -> pd.DataFrame:
    common_ids = target_df.index.intersection(pred_df.index)

    if keep_ids is not None:
        keep_ids = pd.Index(keep_ids).astype(str)
        common_ids = common_ids.intersection(keep_ids)

    return pd.DataFrame(
        {
            "true": pd.to_numeric(
                target_df.loc[common_ids, target_col],
                errors="coerce",
            ),
            "pred": pd.to_numeric(
                pred_df.loc[common_ids, pred_col],
                errors="coerce",
            ),
        }
    ).dropna()


def _score_eval_df(eval_df: pd.DataFrame) -> dict:
    if len(eval_df) < 2:
        return {
            "n": len(eval_df),
            "r2": float("nan"),
            "pearson_r": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "sdep": float("nan"),
        }

    err = eval_df["pred"] - eval_df["true"]
    bias = err.mean()

    return {
        "n": int(len(eval_df)),
        "r2": float(r2_score(eval_df["true"], eval_df["pred"])),
        "pearson_r": float(eval_df["true"].corr(eval_df["pred"], method="pearson")),
        "rmse": float(mean_squared_error(eval_df["true"], eval_df["pred"]) ** 0.5),
        "bias": float(bias),
        "sdep": float((err - bias).pow(2).mean() ** 0.5),
    }


def _plot_true_vs_pred_panel(
    panel_dfs: list[tuple[str, pd.DataFrame, dict]],
    prop: str,
    base_feature: str,
    ft_feature: str,
    save_path: Path,
    dpi: int = 300,
) -> None:
    non_empty_dfs = [eval_df for _, eval_df, _ in panel_dfs if not eval_df.empty]
    if not non_empty_dfs:
        print(f"Skipping plot for {prop} / {base_feature} vs {ft_feature}: no rows.")
        return

    all_vals = pd.concat(
        [df[["true", "pred"]].stack() for df in non_empty_dfs],
        ignore_index=True,
    )
    min_val = all_vals.min()
    max_val = all_vals.max()
    pad = (max_val - min_val) * 0.05 if max_val != min_val else 1.0
    axis_min = min_val - pad
    axis_max = max_val + pad

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    colours = ["tab:blue", "tab:orange", "tab:green"]

    for ax, (label, eval_df, metrics), colour in zip(axes, panel_dfs, colours):
        if eval_df.empty:
            ax.set_title(f"{label}\nNo data")
            ax.axis("off")
            continue

        ax.scatter(
            eval_df["true"],
            eval_df["pred"],
            alpha=0.7,
            edgecolor="black",
            linewidth=0.35,
            color=colour,
        )
        ax.plot(
            [axis_min, axis_max],
            [axis_min, axis_max],
            linestyle="--",
            color="black",
            linewidth=1.0,
        )
        ax.set_xlim(axis_min, axis_max)
        ax.set_ylim(axis_min, axis_max)
        ax.grid(axis="both", linestyle="--", alpha=0.25)
        ax.set_title(
            f"{label}\n"
            f"n={metrics['n']} | R2={metrics['r2']:.3f} | "
            f"r={metrics['pearson_r']:.3f} | RMSE={metrics['rmse']:.3f}",
            fontsize=11,
        )
        ax.set_xlabel("True")

    axes[0].set_ylabel("Predicted")
    fig.suptitle(
        f"{prop}: {base_feature} full vs {base_feature} FT-ID trimmed vs {ft_feature}",
        fontsize=14,
        weight="bold",
    )
    fig.tight_layout()

    save_path.mkdir(parents=True, exist_ok=True)
    safe_base = base_feature.replace("/", "_")
    safe_ft = ft_feature.replace("/", "_")
    fig.savefig(
        save_path / f"{prop}_{safe_base}_trimmed_vs_{safe_ft}_true_vs_pred.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_fine_tune_trimmed_id_comparison(
    properties: list[str] | None = None,
    ft_base_comparisons: list[tuple[str, str]] | None = None,
    save_dir: str | Path | None = None,
) -> pd.DataFrame:
    full_pathing = FULL_PATHING
    preds_dir = full_pathing["prediction_output_dirs"]["rf"]

    if properties is None:
        properties = sorted(preds_dir.keys())

    if ft_base_comparisons is None:
        ft_base_comparisons = FT_BASE_COMPARISONS

    if save_dir is None:
        save_dir = (
            Path(full_pathing["imp_dirs"]["results_dir"])
            / "pp_analysis"
            / "ft_trimmed_id_check"
        )
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for prop in properties:
        for base_feature, ft_feature in ft_base_comparisons:
            base_loaded = _read_target_and_prediction(
                prop=prop,
                feature_set=base_feature,
                full_pathing=full_pathing,
                preds_dir=preds_dir,
                target_columns=TARGET_COLUMNS,
            )
            ft_loaded = _read_target_and_prediction(
                prop=prop,
                feature_set=ft_feature,
                full_pathing=full_pathing,
                preds_dir=preds_dir,
                target_columns=TARGET_COLUMNS,
            )

            if base_loaded is None or ft_loaded is None:
                continue

            target_df, base_pred_df, target_col, base_pred_col = base_loaded
            _, ft_pred_df, _, ft_pred_col = ft_loaded

            ft_eval_full = _make_eval_df(
                target_df=target_df,
                pred_df=ft_pred_df,
                target_col=target_col,
                pred_col=ft_pred_col,
            )
            base_eval_full = _make_eval_df(
                target_df=target_df,
                pred_df=base_pred_df,
                target_col=target_col,
                pred_col=base_pred_col,
            )
            base_eval_trimmed = _make_eval_df(
                target_df=target_df,
                pred_df=base_pred_df,
                target_col=target_col,
                pred_col=base_pred_col,
                keep_ids=ft_eval_full.index,
            )

            panel_items = [
                ("Base full IDs", base_eval_full, _score_eval_df(base_eval_full)),
                (
                    "Base trimmed to FT IDs",
                    base_eval_trimmed,
                    _score_eval_df(base_eval_trimmed),
                ),
                ("Fine-tuned full IDs", ft_eval_full, _score_eval_df(ft_eval_full)),
            ]

            _plot_true_vs_pred_panel(
                panel_dfs=panel_items,
                prop=prop,
                base_feature=base_feature,
                ft_feature=ft_feature,
                save_path=save_dir / prop,
            )

            score_by_panel = {label: metrics for label, _, metrics in panel_items}
            base_full = score_by_panel["Base full IDs"]
            base_trimmed = score_by_panel["Base trimmed to FT IDs"]
            ft_full = score_by_panel["Fine-tuned full IDs"]

            for label, _, metrics in panel_items:
                rows.append(
                    {
                        "property": prop,
                        "base_feature": base_feature,
                        "ft_feature": ft_feature,
                        "feature_pair": f"{base_feature} vs {ft_feature}",
                        "comparison_set": label,
                        **metrics,
                        "delta_r2_vs_base_full": metrics["r2"] - base_full["r2"],
                        "delta_r2_vs_base_trimmed": metrics["r2"]
                        - base_trimmed["r2"],
                        "delta_pearson_r_vs_base_full": metrics["pearson_r"]
                        - base_full["pearson_r"],
                        "delta_pearson_r_vs_base_trimmed": metrics["pearson_r"]
                        - base_trimmed["pearson_r"],
                        "n_base_full": base_full["n"],
                        "n_base_trimmed": base_trimmed["n"],
                        "n_ft_full": ft_full["n"],
                        "n_removed_by_ft_id_trim": base_full["n"]
                        - base_trimmed["n"],
                    }
                )

    summary_df = pd.DataFrame(rows)

    if summary_df.empty:
        print("No fine-tune/base comparison rows were generated.")
        return summary_df

    summary_df.to_csv(
        save_dir / "ft_trimmed_id_comparison_metrics.csv",
        index=False,
    )

    for metric in ["r2", "pearson_r", "rmse"]:
        metric_df = summary_df.dropna(subset=[metric]).copy()
        if metric_df.empty:
            continue

        for feature_pair, pair_df in metric_df.groupby("feature_pair", sort=False):
            safe_pair = feature_pair.replace("/", "_").replace(" ", "_")

            plt.figure(figsize=(max(12, 0.8 * pair_df["property"].nunique()), 7))
            ax = sns.barplot(
                data=pair_df,
                x="property",
                y=metric,
                hue="comparison_set",
                errorbar=None,
            )
            ax.set_title(
                f"Fine-tune ID trimming check: {feature_pair} {metric}",
                weight="bold",
            )
            ax.set_xlabel("property")
            ax.set_ylabel(metric)
            ax.tick_params(axis="x", rotation=45)
            ax.legend(title="")
            plt.tight_layout()
            plt.savefig(
                save_dir / f"ft_trimmed_id_comparison_{safe_pair}_{metric}_bar.png",
                dpi=300,
                bbox_inches="tight",
            )
            plt.close()

    print(f"Saved fine-tune trimmed-ID comparison plots to: {save_dir}")
    print(f"Saved metrics to: {save_dir / 'ft_trimmed_id_comparison_metrics.csv'}")

    return summary_df


# Run all configured property/base/fine-tuned comparisons.
ft_trimmed_id_summary = plot_fine_tune_trimmed_id_comparison()
