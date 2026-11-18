import pandas as pd
from pathlib import Path

# Enable only the analyses you want to run.
run_1 = False  # Fine-tuning training loss
run_2 = False  # Average descriptor performance
run_3 = False  # RDKit feature importance for Mordred autocorrelation
run_4 = False  # Atomic-mass autocorrelation distributions
run_5 = False  # Fine-tuned vs ID-trimmed base predictions
run_6 = False  # Classification ROC/AUC curves
run_7 = False
run_8 = False
run_11 = False  # Mordred group importance across properties
run_9 = False  # MoeType Pearson r SD across descriptors, excluding LabuteASA
run_10 = False  # Property prediction comparison plot
run_12 = False  # Group-balanced Mordred selection across SVR, ridge and fixed-HP RF
run_13 = True

if any((run_1, run_2, run_3, run_4, run_5, run_6, run_7, run_8, run_11)):
    import matplotlib.pyplot as plt

# Try binary descriptors in random order until this many plots succeed.
ROC_PLOT_COUNT = 5
ROC_RANDOM_SEED = None  # Set an integer for a reproducible candidate order.

# %% Plot fine-tuning training loss
if run_1:
    data = pd.read_csv(
        "/users/yhb18174/TL_project/datasets/embeddings/BP_embeddings/"
        "bp_ft-random-molformer-c3-1b_fine_tuned_model/fine_tuning_log_history.csv"
    )

    train_loss = data.dropna(subset=["loss"]).copy()

    x_col = "epoch" if "epoch" in train_loss.columns else "step"

    plt.figure(figsize=(7, 4))
    plt.plot(train_loss[x_col], train_loss["loss"], marker="o")
    plt.xlabel(x_col.capitalize())
    plt.ylabel("Training loss")
    plt.title("Fine-tuning Training Loss")
    plt.tight_layout()
    plt.show()


# %% Checking averages
if run_2:

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
            total_average_df.to_csv(
                exp_dir / "total_descriptor_average.csv", index=False
            )

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

    for exp, exp_dir in cfp_block.items():
        exp_parts = exp.split("_")

        if len(exp_parts) < 4:
            continue

        pred_feature = exp_parts[1]

        if pred_feature != "mordred" and exp != "pred_rdkit_tr_mordred":
            continue

        try:
            total_average_df = averageExperimentPerformanceTotalDescriptors(
                exp=exp,
                exp_dir=exp_dir,
            )
            total_average_dfs.append(total_average_df)
        except Exception as e:
            print(e)

    all_total_average_df = pd.concat(total_average_dfs, ignore_index=True)
    numeric_cols = all_total_average_df.select_dtypes(include="number").columns
    all_total_average_df[numeric_cols] = all_total_average_df[numeric_cols].round(3)
    all_total_average_df.to_csv(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_all_task_metric_avg_perf.csv",
        index=False,
    )


# %% Plot top RDKit features for predicting Mordred Autocorrelation descriptors
if run_3:
    import sys

    CONFIG_DIR = Path(__file__).resolve().parents[2] / "run" / "config"
    sys.path.insert(0, str(CONFIG_DIR))
    from config import FULL_PATHING, SRC_DIR

    cfp_block = FULL_PATHING["prediction_output_dirs"][
        "lipinski_cross_feature_predictions"
    ]["all"]
    sys.path.insert(0, str(SRC_DIR / "visualisation"))
    from vis import Visualise

    sys.path.insert(0, str(SRC_DIR / "datasets"))
    from group_descriptors import getMordredGroups

    # Get all Mordred descriptors belonging to Autocorrelation
    res = getMordredGroups()
    autocorrelation_descs = res["Autocorrelation"]

    v = Visualise(save_all=False)

    original_data = pd.read_csv(
        cfp_block["pred_mordred_tr_rdkit"] / "all_feature_importance.csv"
    )

    # Corresponding importance columns
    cols_to_avg = [
        f"Importance_{desc}"
        for desc in autocorrelation_descs
        if f"Importance_{desc}" in original_data.columns
    ]

    # Mean importance of each RDKit feature across all
    # Autocorrelation Mordred descriptors
    original_data["Importance_Autocorrelation"] = original_data[cols_to_avg].mean(
        axis=1
    )

    v.plotFeatureImportance(
        data=original_data,
        x_col="Importance_Autocorrelation",
        y_col="Feature",
        top_n=25,
        save_path=str(Path(__file__).parent),
    )


# %% Plot Mordred atomic-mass autocorrelation descriptor distributions
if run_4:
    import sys

    CONFIG_DIR = Path(__file__).resolve().parents[2] / "run" / "config"
    sys.path.insert(0, str(CONFIG_DIR))
    from config import FULL_PATHING, SRC_DIR

    import math
    import re

    import numpy as np

    FILE_DIR = Path(__file__).resolve().parent
    PROJ_DIR = FILE_DIR.parents[1]
    AUTOCORR_FAMILIES = ["ATS", "AATS", "ATSC", "AATSC", "MATS", "GATS"]
    PROPERTY_SUFFIX = "m"
    TRAINING_PERCENTILE = 0.99

    mordred_features_path = Path(FULL_PATHING["full_features"]["all"]["mordred"])
    if not mordred_features_path.exists():
        mordred_features_path = (
            PROJ_DIR
            / "run"
            / "test"
            / "expected_test_results"
            / "expected_mordred_features.csv"
        )

    mordred_features = pd.read_csv(mordred_features_path, index_col=0, low_memory=False)

    print(f"Loaded Mordred features from: {mordred_features_path}")

    for family in AUTOCORR_FAMILIES:
        mass_pattern = re.compile(
            rf"^{family}(?P<lag>\d+){PROPERTY_SUFFIX}(?:_mordred)?$"
        )

        mass_cols = []
        for col in mordred_features.columns:
            match = mass_pattern.match(col)
            if match:
                mass_cols.append((int(match.group("lag")), col))

        mass_cols = [col for _, col in sorted(mass_cols)]

        if not mass_cols:
            print(f"Skipping {family}: no atomic-mass descriptors found.")
            continue

        mass_values = []
        percentile_cutoffs = []
        n_raw_values = []

        for col in mass_cols:
            non_nan_values = pd.to_numeric(
                mordred_features[col], errors="coerce"
            ).dropna()
            percentile_cutoff = non_nan_values.quantile(TRAINING_PERCENTILE)
            trained_values = non_nan_values.loc[non_nan_values <= percentile_cutoff]

            mass_values.append(trained_values)
            percentile_cutoffs.append(percentile_cutoff)
            n_raw_values.append(len(non_nan_values))

        all_mass_values = pd.concat(mass_values, ignore_index=True)

        if all_mass_values.empty:
            print(
                f"Skipping {family}: descriptors found, but no values remained after "
                f"the {TRAINING_PERCENTILE:.0%} cutoff."
            )
            continue

        bin_min = all_mass_values.min()
        bin_max = all_mass_values.max()
        if bin_min == bin_max:
            bin_min -= 0.5
            bin_max += 0.5

        bin_edges = np.linspace(bin_min, bin_max, 41)
        n_plot_cols = 3
        n_plot_rows = math.ceil(len(mass_cols) / n_plot_cols)

        fig, axes = plt.subplots(
            n_plot_rows,
            n_plot_cols,
            figsize=(4.8 * n_plot_cols, 3.6 * n_plot_rows),
            squeeze=False,
        )
        axes_flat = axes.ravel()

        for ax, col, trained_values, percentile_cutoff, n_raw in zip(
            axes_flat,
            mass_cols,
            mass_values,
            percentile_cutoffs,
            n_raw_values,
        ):
            pct_zero = 100 * (trained_values == 0).mean()
            pct_removed = 100 * (1 - (len(trained_values) / n_raw)) if n_raw else 0
            ax.hist(
                trained_values,
                bins=bin_edges,
                edgecolor="black",
                color="steelblue",
                alpha=0.85,
            )
            ax.set_title(
                f"{col}\ntrain n={len(trained_values)}, 0={pct_zero:.1f}%, "
                f"cut>{TRAINING_PERCENTILE:.0%} ({pct_removed:.1f}%)",
                fontsize=11,
            )
            ax.set_xlabel(f"Descriptor value <= p99 ({percentile_cutoff:.3g})")
            ax.set_ylabel("Count")
            ax.grid(axis="y", linestyle="--", alpha=0.25)

        for ax in axes_flat[len(mass_cols) :]:
            ax.axis("off")

        fig.suptitle(
            f"Mordred Autocorrelation: {family} Atomic Mass",
            fontsize=14,
            weight="bold",
        )
        fig.tight_layout()

        save_path = FILE_DIR / f"{family.lower()}_mass_histograms.png"
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close(fig)

        print(f"Plotted {family} descriptors: {', '.join(mass_cols)}")
        print(f"Saved histogram plot to: {save_path}")


# %% Compare fine-tuned predictions against ID-trimmed base predictions
if run_5:
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
            print(
                f"Skipping plot for {prop} / {base_feature} vs {ft_feature}: no rows."
            )
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


# %% Plot ROC/AUC curves for binary and multiclass Mordred descriptors
if run_6:
    # Enable run_6 above to run this analysis without the other experiments.
    import sys
    import re
    import shutil
    import tarfile
    from glob import glob
    from pathlib import Path

    import joblib
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    _PROJECT_DIR = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_PROJECT_DIR / "run" / "config"))
    sys.path.insert(0, str(_PROJECT_DIR / "scripts" / "src" / "datasets"))
    from config import FULL_PATHING
    from feature_cleaning import cleanFeatureDF

    CLASSIFICATION_EXPERIMENT_DIR = Path(
        "/users/yhb18174/TL_project/results/"
        "lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_rdkit"
    )

    def plot_mordred_classification_roc(
        experiment_dir=CLASSIFICATION_EXPERIMENT_DIR,
        rdkit_path=None,
        mordred_path=None,
        show=False,
        plot_count=5,
        random_seed=None,
    ):
        """Save ROC plots excluding every molecule listed in training_ids.csv.

        Input feature paths may be glob patterns, as in the pipeline configuration.
        Only classification models are extracted. Load only your trusted archive.
        Try randomly ordered binary targets until plot_count plots succeed.
        Missing models and other failed candidates are skipped.
        Each model still receives its complete, ordered set of RDKit predictors.
        The exclusion file must cover training IDs for every selected model;
        this function cannot establish that provenance from the model alone.
        """
        from datetime import datetime
        from time import perf_counter

        started = perf_counter()

        def update(message):
            elapsed = perf_counter() - started
            print(
                f"[{datetime.now():%Y-%m-%d %H:%M:%S}] [ROC +{elapsed:.1f}s] {message}",
                flush=True,
            )

        experiment_dir = Path(experiment_dir)
        if not isinstance(plot_count, int) or plot_count < 1:
            raise ValueError("plot_count must be a positive integer")
        update(
            f"Reading descriptor results: {experiment_dir / 'pred_mordred_tr_rdkit.csv'}"
        )
        results = pd.read_csv(experiment_dir / "pred_mordred_tr_rdkit.csv", index_col=0)
        results.index = results.index.astype(str).str.strip()
        if not results.index.is_unique:
            raise ValueError("Descriptor results contain duplicate names")
        classification = results.loc[
            results["task_type"] == "binary_classification"
        ].sample(frac=1, random_state=random_seed)
        update(
            f"Trying binary descriptors in random order until {plot_count} plots succeed"
        )
        output_dir = experiment_dir / "classification_roc" / "training_ids_excluded"
        output_dir.mkdir(parents=True, exist_ok=True)
        classification.to_csv(output_dir / "classification_descriptors.csv")
        update(
            f"Found {len(classification)} classification descriptors: "
            f"{classification['task_type'].value_counts().to_dict()}. Output: {output_dir}"
        )
        if classification.empty:
            update("No classification descriptors found.")
            return pd.DataFrame()

        training_ids_path = experiment_dir / "training_data" / "training_ids.csv"
        update(f"Reading mandatory training-ID exclusions: {training_ids_path}")
        training_id_frame = pd.read_csv(
            training_ids_path, dtype=str, keep_default_na=False
        )
        if "ID" not in training_id_frame.columns:
            raise ValueError(
                f"Training ID file must contain an ID column: {training_ids_path}"
            )
        training_id_values = training_id_frame["ID"].str.strip()
        if training_id_values.empty or training_id_values.eq("").any():
            raise ValueError(
                "Training ID file is empty or contains blank IDs; refusing to evaluate"
            )
        training_ids = pd.Index(training_id_values.unique())
        update(f"Loaded {len(training_ids):,} unique training IDs to exclude")

        def load_features(path, exclude_training=True):
            update(f"Finding feature files: {path}")
            files = sorted(glob(str(path)))
            if not files and str(path).endswith("_*.csv"):
                files = sorted(glob(str(path).replace("_*.csv", ".csv")))
            if not files:
                raise FileNotFoundError(f"No feature files matched {path}")
            frames = []
            for file_number, p in enumerate(files, 1):
                update(f"Reading feature file {file_number}/{len(files)}: {p}")
                frames.append(
                    pd.read_csv(p, index_col="ID", dtype={"ID": str}, low_memory=False)
                )
                update(
                    f"Loaded {frames[-1].shape[0]:,} rows, {frames[-1].shape[1]:,} columns"
                )
            frame = pd.concat(frames)
            del frames
            if frame.index.isna().any():
                raise ValueError(f"Missing molecule IDs in {path}")
            frame.index = frame.index.astype(str).str.strip()
            if (frame.index == "").any():
                raise ValueError(f"Blank molecule IDs in {path}")
            if not frame.index.is_unique:
                raise ValueError(f"Duplicate molecule IDs in {path}")
            if exclude_training:
                excluded = frame.index.isin(training_ids)
                frame = frame.loc[~excluded].copy()
                update(
                    f"Excluded {int(excluded.sum()):,} training molecules from {path}; "
                    f"{len(frame):,} rows remain"
                )
                if frame.empty:
                    raise ValueError(
                        f"No evaluation molecules remain after training-ID exclusion in {path}"
                    )
            # Retain constant/high-missing columns: the saved model, rather
            # than a new feature-selection pass, determines its input schema.
            # Do not infer imputation values from evaluation data or impute
            # true labels. Invalid rows are removed per descriptor below.
            update(
                f"Cleaning features: {frame.shape[0]:,} rows, {frame.shape[1]:,} columns"
            )
            frame, _ = cleanFeatureDF(
                frame,
                max_nan_fraction=1.0,
                drop_constant_cols=False,
                median_impute=False,
                correlation_threshold=None,
            )
            update(
                f"Cleaning complete: {frame.shape[0]:,} rows, {frame.shape[1]:,} columns"
            )
            return frame

        # Evaluate on the full dataset, excluding the saved training IDs.
        configured = FULL_PATHING["full_features"]["all"]
        features = load_features(
            rdkit_path if rdkit_path is not None else configured["rdkit"]
        )
        targets = load_features(
            mordred_path if mordred_path is not None else configured["mordred"],
            exclude_training=False,
        )
        training_targets = targets
        common_ids = features.index.intersection(targets.index)
        if common_ids.empty:
            raise ValueError(
                "No common evaluation IDs remain after excluding training molecules"
            )
        features, targets = features.loc[common_ids], targets.loc[common_ids]
        update(f"Aligned RDKit and Mordred data: {len(common_ids):,} molecule IDs")
        model_dir = output_dir / "extracted_models"
        model_dir.mkdir(exist_ok=True)
        metric_rows, status_rows, class_count_rows = [], [], []
        plotted = 0
        archive_path = experiment_dir / "training_data" / "models.tar.gz"

        update(f"Scanning model archive: {archive_path}")
        with tarfile.open(archive_path, "r:gz") as archive:
            members = {}
            for member_number, member in enumerate(archive, 1):
                if member.isfile():
                    members.setdefault(Path(member.name).name, []).append(member)
                if member_number % 100 == 0:
                    update(f"Scanned {member_number:,} archive entries")
            update(
                f"Archive scan complete: {sum(map(len, members.values())):,} regular files"
            )
            for number, (descriptor, result) in enumerate(classification.iterrows(), 1):
                if plotted >= plot_count:
                    break
                descriptor = str(descriptor).strip()
                descriptor_started = perf_counter()
                update(
                    f"[{number}/{len(classification)}] Starting {descriptor} ({result['task_type']})"
                )
                fig = None
                try:
                    matches = members.get(f"{descriptor}_model.pkl.gz", [])
                    if not matches:
                        status_rows.append(
                            dict(
                                descriptor=descriptor,
                                status="missing_model",
                                error="No matching model in archive",
                            )
                        )
                        update(
                            f"SKIPPING {descriptor}: model missing; trying another binary descriptor"
                        )
                        continue
                    if len(matches) != 1:
                        raise ValueError(
                            f"Expected one model in archive; found {len(matches)}"
                        )
                    # Copy a regular member to a generated filename; archive paths
                    # and links cannot escape the extraction directory.
                    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", descriptor)
                    stem = f"{number:04d}_{safe_name}"
                    model_path = model_dir / f"{stem}_model.pkl.gz"
                    update(
                        f"Extracting {matches[0].name} ({matches[0].size / 1024**2:.1f} MiB)"
                    )
                    with archive.extractfile(matches[0]) as source, model_path.open(
                        "wb"
                    ) as dest:
                        shutil.copyfileobj(source, dest)
                    update(f"Loading model: {model_path}")
                    model = joblib.load(model_path)
                    update(
                        "Model loaded; aligning predictors and filtering evaluation rows"
                    )
                    if not hasattr(model, "feature_names_in_"):
                        raise ValueError(
                            "Model lacks feature names; cannot verify predictor order"
                        )
                    required_features = list(model.feature_names_in_)
                    missing_features = [
                        name
                        for name in required_features
                        if name not in features.columns
                    ]
                    if missing_features:
                        raise ValueError(
                            f"Model-required RDKit descriptors are absent from the source data: {missing_features}. "
                            "Provide rdkit_path pointing to the feature dataset used for training."
                        )
                    x = features.loc[:, required_features]
                    update(
                        f"Selected all {len(required_features)} model-trained RDKit inputs in training order"
                    )
                    y = targets[descriptor]
                    if pd.api.types.is_numeric_dtype(y):
                        y = y.round()
                    classes = np.asarray(model.classes_)
                    expected_binary = result["task_type"] == "binary_classification"
                    if (expected_binary and len(classes) != 2) or (
                        not expected_binary and len(classes) < 3
                    ):
                        raise ValueError("Model classes disagree with CSV task_type")
                    train_ids_for_target = training_targets.index.intersection(
                        training_ids
                    )
                    train_y = training_targets.loc[train_ids_for_target, descriptor]
                    if pd.api.types.is_numeric_dtype(train_y):
                        train_y = train_y.round()
                    train_y = train_y.loc[train_y.isin(classes)]
                    train_counts = train_y.value_counts(dropna=False)
                    # Training excluded rare classes; only score model-supported labels.
                    # sklearn forests convert predictors to float32. Values
                    # can be finite in float64 yet overflow that conversion.
                    input_values = x.to_numpy(dtype=np.float64)
                    float32_limit = np.finfo(np.float32).max
                    finite_values = np.isfinite(input_values)
                    oversized_values = finite_values & (
                        np.abs(input_values) > float32_limit
                    )
                    valid_input_rows = (finite_values & ~oversized_values).all(axis=1)
                    if oversized_values.any():
                        affected_columns = x.columns[
                            oversized_values.any(axis=0)
                        ].tolist()
                        update(
                            f"Excluding {int(oversized_values.any(axis=1).sum()):,} rows "
                            f"with inputs outside float32 range; descriptors: {affected_columns}"
                        )
                    update(
                        f"Non-finite input rows: {int((~finite_values.all(axis=1)).sum()):,}"
                    )
                    keep = y.isin(classes) & valid_input_rows
                    update(
                        f"Keeping {int(keep.sum()):,}/{len(keep):,} rows with supported labels "
                        "and finite, float32-representable inputs"
                    )
                    x, y = x.loc[keep], y.loc[keep]
                    eval_counts = y.value_counts(dropna=False)
                    for cls in classes:
                        class_count_rows.append(
                            dict(
                                descriptor=descriptor,
                                task_type=result["task_type"],
                                class_label=cls,
                                training_count=int(train_counts.get(cls, 0)),
                                test_count=int(eval_counts.get(cls, 0)),
                                training_total=int(train_counts.sum()),
                                test_total=int(eval_counts.sum()),
                                evaluation="training_ids_excluded",
                                status="ok",
                            )
                        )
                    update(
                        "Class counts after filtering: "
                        f"training={train_counts.to_dict()}, test={eval_counts.to_dict()}"
                    )
                    x = x.astype(np.float32)
                    if x.index.isin(training_ids).any():
                        raise RuntimeError(
                            "Training-ID exclusion failed; refusing to predict"
                        )
                    if y.nunique() < 2:
                        raise ValueError("Fewer than two classes available for ROC")
                    update(
                        f"Predicting probabilities for {len(y):,} rows, "
                        f"{x.shape[1]:,} predictors, classes={classes.tolist()}"
                    )
                    probabilities = np.asarray(model.predict_proba(x))
                    update("Prediction complete; calculating ROC curves")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    descriptor_rows = []
                    indices = [1] if expected_binary else range(len(classes))
                    for j in indices:
                        binary_y = (y.to_numpy() == classes[j]).astype(int)
                        if np.unique(binary_y).size < 2:
                            update(
                                f"Class {classes[j]}: skipping ROC (class absent or no negatives)"
                            )
                            descriptor_rows.append(
                                dict(
                                    descriptor=descriptor,
                                    task_type=result["task_type"],
                                    class_label=classes[j],
                                    auc=np.nan,
                                    n=len(y),
                                    evaluation="training_ids_excluded",
                                    status="class absent or no negatives",
                                )
                            )
                            continue
                        fpr, tpr, _ = roc_curve(binary_y, probabilities[:, j])
                        score = auc(fpr, tpr)
                        update(f"Class {classes[j]}: AUC={score:.4f}")
                        ax.plot(
                            fpr, tpr, label=f"Class {classes[j]} (AUC = {score:.3f})"
                        )
                        descriptor_rows.append(
                            dict(
                                descriptor=descriptor,
                                task_type=result["task_type"],
                                class_label=classes[j],
                                auc=score,
                                n=len(y),
                                evaluation="training_ids_excluded",
                                status="ok",
                            )
                        )
                    macro = np.mean([row["auc"] for row in descriptor_rows])
                    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
                    ax.set(
                        xlim=(0, 1),
                        ylim=(0, 1.02),
                        xlabel="False positive rate",
                        ylabel="True positive rate",
                        title=f"{descriptor}: {'binary' if expected_binary else 'one-vs-rest'} ROC\n"
                        f"Listed training IDs excluded; n={len(y)}"
                        + (f"; macro AUC={macro:.3f}" if not expected_binary else ""),
                    )
                    ax.legend(loc="lower right")
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    plot_path = output_dir / f"{stem}_roc.png"
                    update(f"Saving plot: {plot_path}")
                    fig.savefig(plot_path, dpi=300, bbox_inches="tight")
                    pd.DataFrame({"ID": x.index}).to_csv(
                        output_dir / f"{stem}_evaluation_ids.csv",
                        index=False,
                    )
                    if show:
                        plt.show()
                    metric_rows.extend(descriptor_rows)
                    status_rows.append(
                        dict(descriptor=descriptor, status="plotted", error="")
                    )
                    plotted += 1
                    update(
                        f"[{number}/{len(classification)}] Plotted {descriptor} "
                        f"in {perf_counter() - descriptor_started:.1f}s; "
                        f"{plotted}/{plot_count} requested plots complete"
                    )
                except Exception as exc:
                    status_rows.append(
                        dict(descriptor=descriptor, status="failed", error=str(exc))
                    )
                    update(
                        f"[{number}/{len(classification)}] FAILED {descriptor}: {exc}"
                    )
                finally:
                    if fig is not None:
                        plt.close(fig)

        metrics = pd.DataFrame(metric_rows)
        update("Saving AUC metrics and descriptor status CSVs")
        metrics.to_csv(output_dir / "roc_auc_metrics.csv", index=False)
        pd.DataFrame(class_count_rows).to_csv(
            output_dir / "roc_class_counts.csv", index=False
        )
        pd.DataFrame(status_rows).to_csv(output_dir / "plot_status.csv", index=False)
        plotted = sum(row["status"] == "plotted" for row in status_rows)
        selected_names = [
            row["descriptor"] for row in status_rows if row["status"] == "plotted"
        ]
        classification.loc[selected_names].to_csv(
            output_dir / "plotted_descriptors.csv"
        )
        if plotted < plot_count:
            update(
                f"Candidate list exhausted: only {plotted} of {plot_count} requested plots could be generated"
            )
        update(
            f"Finished: {plotted}/{plot_count} requested descriptors plotted, "
            f"{len(status_rows) - plotted} candidates skipped or failed. "
            f"Saved {len(metric_rows)} class AUC records and plot status to {output_dir}"
        )
        return metrics

    classification_roc_summary = plot_mordred_classification_roc(
        plot_count=ROC_PLOT_COUNT,
        random_seed=ROC_RANDOM_SEED,
    )


if run_7:
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    from pathlib import Path
    from mordred import Calculator, descriptors

    def getMordredGroups(prefix="_mordred"):
        """Return descriptor groups for Mordred descriptors."""
        groups = {}
        calc = Calculator(descriptors, ignore_3D=False)

        for desc in calc.descriptors:
            module = desc.__module__.split(".")[-1]
            name = str(desc).split(".")[-1]
            name = f"{name}{prefix}"  # apply prefix
            groups.setdefault(module, []).append(name)

        return groups

    csv_path = Path(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_maccs/ridge/pred_mordred_tr_maccs.csv"
    )
    metrics = ["RMSE", "r2", "Pearson_r"]  # any metric that has a matching *_std column
    task = "regression"
    bins = 50
    threshold = 5.0  # % SD above which a descriptor is saved to the CSV

    # Built once (the Calculator is slow to construct)
    groups = getMordredGroups(prefix="_mordred")
    desc_to_group = {d: g for g, members in groups.items() for d in members}

    perf = pd.read_csv(csv_path, index_col=0)
    if "task_type" in perf.columns:
        perf = perf.loc[perf["task_type"] == task]

    for metric in metrics:
        std_col = f"{metric}_std"
        if metric not in perf.columns or std_col not in perf.columns:
            print(f"Skipping {metric}: missing '{metric}' or '{std_col}'")
            continue

        res = pd.DataFrame(
            {
                "mean": pd.to_numeric(perf[metric], errors="coerce"),
                "std": pd.to_numeric(perf[std_col], errors="coerce"),
            }
        ).dropna()
        res["pct_std"] = res["std"] / res["mean"].abs() * 100
        res.index.name = "descriptor"
        res = res.replace(-np.inf, np.inf).dropna(subset=["pct_std"])
        if res.empty:
            print(f"Skipping {metric}: no usable values (n_resamples=1 gives NaN std)")
            continue

        # ---- CSV of descriptors above the threshold, highest to lowest ----
        high = res.loc[res["pct_std"] > threshold].sort_values(
            "pct_std", ascending=False
        )
        high = high.rename(columns={"mean": f"{metric}_mean", "std": f"{metric}_std"})
        high_path = (
            csv_path.parent / f"{csv_path.stem}_{metric}_pct_std_over_{threshold:g}.csv"
        )
        high.to_csv(high_path)
        print(f"{len(high)}/{len(res)} descriptors > {threshold:g}% -> {high_path}")

        # ---- Histogram of everything ----
        pct = res["pct_std"]
        n_zero = int((pct == 0).sum())
        n_inf = int(np.isinf(pct).sum())
        positive = pct[np.isfinite(pct) & (pct > 0)]

        if positive.empty:
            print(f"Skipping histogram for {metric}: no positive finite values")
        else:
            lo, hi = positive.min(), positive.max()
            if lo == hi:  # avoid a zero-width log range
                lo, hi = lo / 1.5, hi * 1.5
            edges = np.logspace(np.log10(lo), np.log10(hi), bins + 1)
            step = edges[1] / edges[0]  # bin width as a ratio (log space)
            median = pct[np.isfinite(pct)].median()

            fig, ax = plt.subplots(figsize=(10, 5))
            counts, _ = np.histogram(positive, bins=edges)
            ax.bar(
                edges[:-1],
                counts,
                width=np.diff(edges),
                align="edge",
                color="#4f8a8b",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )

            extra_ticks, extra_labels = [], []
            if n_zero:
                x0 = edges[0] / step**3
                ax.bar(
                    x0,
                    n_zero,
                    width=x0 * (step - 1),
                    align="edge",
                    color="#7a7a7a",
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.85,
                )
                ax.text(
                    x0 * np.sqrt(step), n_zero, str(n_zero), ha="center", va="bottom"
                )
                extra_ticks.append(x0 * np.sqrt(step))
                extra_labels.append("0%")
            if n_inf:
                x1 = edges[-1] * step**2
                ax.bar(
                    x1,
                    n_inf,
                    width=x1 * (step - 1),
                    align="edge",
                    color="#b94a48",
                    edgecolor="black",
                    linewidth=0.6,
                    alpha=0.85,
                )
                ax.text(x1 * np.sqrt(step), n_inf, str(n_inf), ha="center", va="bottom")
                extra_ticks.append(x1 * np.sqrt(step))
                extra_labels.append("∞\n(zero mean)")

            ax.set_xscale("log")
            d_lo, d_hi = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
            decades = [
                10.0**d
                for d in range(d_lo, d_hi + 1)
                if lo / step <= 10.0**d <= hi * step
            ]
            ax.xaxis.set_minor_locator(NullLocator())
            ax.set_xticks(decades + extra_ticks)
            ax.set_xticklabels([f"{d:g}" for d in decades] + extra_labels)

            ax.axvline(
                threshold,
                color="black",
                linestyle=":",
                linewidth=1.5,
                label=f"{threshold:g}% threshold ({len(high)} above)",
            )
            ax.axvline(
                median,
                color="#b94a48",
                linestyle="--",
                linewidth=1.5,
                label=f"Median = {median:.2f}%",
            )
            ax.set_xlabel(f"{metric} SD as % of mean (across resamples, log scale)")
            ax.set_ylabel("Number of descriptors")
            ax.set_title(f"{csv_path.stem}: {metric} variability (n={len(res)})")
            ax.legend()
            ax.grid(axis="y", linestyle="--", alpha=0.35)
            fig.tight_layout()

            out = csv_path.parent / f"{csv_path.stem}_{metric}_pct_std_hist.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out}")

        # ---- Histogram on a linear scale (full range + zoomed) ----
        zoom_pct = 95  # right panel shows values up to this percentile
        pct_all = res["pct_std"]
        finite_pct = pct_all[np.isfinite(pct_all)]  # includes 0% SD descriptors
        n_inf_lin = int(np.isinf(pct_all).sum())

        if finite_pct.empty:
            print(f"Skipping linear histogram for {metric}: no finite values")
        else:
            median_lin = finite_pct.median()
            zoom_max = np.percentile(finite_pct, zoom_pct)
            zoomed = finite_pct[finite_pct <= zoom_max]

            fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(14, 5))

            ax_full.hist(
                finite_pct,
                bins=bins,
                color="#4f8a8b",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
            ax_full.set_title(f"All finite values (n={len(finite_pct)})")

            ax_zoom.hist(
                zoomed,
                bins=bins,
                color="#4f8a8b",
                edgecolor="black",
                linewidth=0.6,
                alpha=0.85,
            )
            ax_zoom.set_title(
                f"Zoomed to {zoom_pct}th percentile (≤ {zoom_max:.1f}%; "
                f"{len(finite_pct) - len(zoomed)} values not shown)"
            )

            for ax, xmax in ((ax_full, finite_pct.max()), (ax_zoom, zoom_max)):
                if threshold <= xmax:
                    ax.axvline(
                        threshold,
                        color="black",
                        linestyle=":",
                        linewidth=1.5,
                        label=f"{threshold:g}% threshold",
                    )
                if median_lin <= xmax:
                    ax.axvline(
                        median_lin,
                        color="#b94a48",
                        linestyle="--",
                        linewidth=1.5,
                        label=f"Median = {median_lin:.2f}%",
                    )
                ax.set_xlabel(f"{metric} SD as % of mean (across resamples)")
                ax.set_ylabel("Number of descriptors")
                ax.grid(axis="y", linestyle="--", alpha=0.35)
                ax.legend()

            suptitle = f"{csv_path.stem}: {metric} variability (linear scale)"
            if n_inf_lin:
                suptitle += f"\n{n_inf_lin} descriptors with zero mean (∞%) not shown"
            fig.suptitle(suptitle)
            fig.tight_layout()

            out = csv_path.parent / f"{csv_path.stem}_{metric}_pct_std_hist_linear.png"
            fig.savefig(out, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {out}")

        # ---- SD % by Mordred descriptor group ----
        gdf = res.copy()
        gdf["group"] = [desc_to_group.get(d, "Unmapped") for d in gdf.index]
        n_unmapped = int((gdf["group"] == "Unmapped").sum())
        if n_unmapped:
            print(
                f"{n_unmapped} descriptors not matched to a group "
                f"(check the prefix matches the CSV index, e.g. '_mordred')"
            )

        rows = {}
        for g, sub in gdf.groupby("group"):
            p = sub["pct_std"]
            fin = p[np.isfinite(p)]
            rows[g] = {
                "n": len(p),
                "median_pct_std": fin.median() if len(fin) else np.nan,
                "mean_pct_std": fin.mean() if len(fin) else np.nan,
                "max_pct_std": p.max(),
                f"n_over_{threshold:g}pct": int((p > threshold).sum()),
                "n_zero_sd": int((p == 0).sum()),
                "n_inf": int(np.isinf(p).sum()),
            }
        group_summary = (
            pd.DataFrame.from_dict(rows, orient="index")
            .rename_axis("group")
            .sort_values("mean_pct_std", ascending=False)
        )
        group_csv = csv_path.parent / f"{csv_path.stem}_{metric}_pct_std_by_group.csv"
        group_summary.to_csv(group_csv)
        print(f"Saved: {group_csv}")

        plot_df = gdf[np.isfinite(gdf["pct_std"]) & (gdf["pct_std"] > 0)]
        if plot_df.empty:
            print(f"Skipping group plot for {metric}: no positive finite values")
            continue

        order = (
            plot_df.groupby("group")["pct_std"]
            .mean()
            .sort_values(ascending=True)
            .index.tolist()
        )
        data = [plot_df.loc[plot_df["group"] == g, "pct_std"].to_numpy() for g in order]

        ylabels = []
        for g in order:
            s = group_summary.loc[g]
            extra = []
            if s["n_zero_sd"]:
                extra.append(f"{int(s['n_zero_sd'])}×0%")
            if s["n_inf"]:
                extra.append(f"{int(s['n_inf'])}×∞")
            ylabels.append(
                f"{g} (n={int(s['n'])}"
                + (f"; {', '.join(extra)}" if extra else "")
                + ")"
            )

        fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(order) + 1.5)))
        ax.boxplot(
            data,
            vert=False,
            positions=range(len(order)),
            widths=0.6,
            showfliers=False,
            patch_artist=True,
            boxprops={"facecolor": "#4f8a8b", "alpha": 0.5},
            medianprops={"color": "#b94a48", "linewidth": 1.5},
        )
        rng = np.random.default_rng(0)
        for i, vals in enumerate(data):
            ax.scatter(
                vals,
                i + rng.uniform(-0.2, 0.2, len(vals)),
                s=8,
                color="black",
                alpha=0.5,
                zorder=3,
            )

        ax.set_xscale("log")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.axvline(
            threshold,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"{threshold:g}% threshold",
        )
        ax.set_xlabel(f"{metric} SD as % of mean (across resamples, log scale)")
        ax.set_title(f"{csv_path.stem}: {metric} variability by Mordred group")
        ax.legend(loc="lower right")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        fig.tight_layout()

        out = csv_path.parent / f"{csv_path.stem}_{metric}_pct_std_by_group.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out}")


if run_8:
    import numpy as np
    import pandas as pd

    std_path = "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/"
    paths = [
        "pred_rdkit_tr_mordred/pred_rdkit_tr_mordred.csv",
        # "pred_rdkit_tr_maccs/pred_rdkit_tr_maccs.csv",
        # "pred_rdkit_tr_morgan/pred_rdkit_tr_morgan.csv",
        "pred_mordred_tr_rdkit/random_forest/pred_mordred_tr_rdkit.csv",
        "pred_mordred_tr_maccs/pred_mordred_tr_maccs.csv",
        "pred_mordred_tr_morgan/pred_mordred_tr_morgan.csv",
        "pred_mordred_tr_chemberta-dc/random_forest/pred_mordred_tr_chemberta-dc.csv",
        "pred_mordred_tr_chemberta-sey/random_forest/pred_mordred_tr_chemberta-sey.csv",
        "pred_mordred_tr_molformer-ibm/random_forest/pred_mordred_tr_molformer-ibm.csv",
        "pred_mordred_tr_molformer-dc/random_forest/pred_mordred_tr_molformer-dc.csv",
        "pred_mordred_tr_selformer/random_forest/pred_mordred_tr_selformer.csv",
    ]

    paths = [
        # "pred_rdkit_tr_mordred/pred_rdkit_tr_mordred.csv",
        # "pred_rdkit_tr_maccs/pred_rdkit_tr_maccs.csv",
        # "pred_rdkit_tr_morgan/pred_rdkit_tr_morgan.csv",
        # "pred_mordred_tr_rdkit/pred_mordred_tr_rdkit.csv",
        # "pred_mordred_tr_maccs/pred_mordred_tr_maccs.csv",
        # "pred_mordred_tr_morgan/pred_mordred_tr_morgan.csv",
        # "pred_mordred_tr_chemberta/pred_mordred_tr_chemberta.csv",
        # "pred_mordred_tr_chembertasey/pred_mordred_tr_chembertasey.csv",
        # "pred_mordred_tr_molformer/pred_mordred_tr_molformer.csv",
        # "pred_mordred_tr_molformer-c3-1b/pred_mordred_tr_molformer-c3-1b.csv",
        # "pred_mordred_tr_selformer/pred_mordred_tr_selformer.csv",
        "pred_mordred_tr_rdkit/random_forest/regression_group_perf.csv",
        "pred_mordred_tr_maccs/regression_group_perf.csv",
        "pred_mordred_tr_morgan/regression_group_perf.csv",
        "pred_mordred_tr_chemberta-dc/analysis/random_forest/regression_group_perf.csv",
        "pred_mordred_tr_chemberta-sey/analysis/random_forest/regression_group_perf.csv",
        "pred_mordred_tr_molformer-ibm/analysis/random_forest/regression_group_perf.csv",
        "pred_mordred_tr_molformer-dc/analysis/random_forest/regression_group_perf.csv",
        "pred_mordred_tr_selformer/analysis/random_forest/regression_group_perf.csv",
        "pred_mordred_avg_transformers/group_performance_by_task.csv",
    ]

    groups = [
        "RingCount",
        "Framework",
        "AtomCount",
        "RotatableBond",
        "TopologicalIndex",
        "Chi",
        "KappaShapeIndex",
        "WalkCount",
        "PathCount",
        "MolecularDistanceEdge",
        "GeometricalIndex",
        "PBF",
        "MoRSE",
    ]

    plot_rows = []
    for path in paths:
        csv_path = Path(std_path) / path
        print(f"\n{csv_path}")
        df = pd.read_csv(csv_path, index_col=0)
        # Group names are index values; task type, when present, is a column.
        if "task_type" in df.columns:
            df = df.loc[df["task_type"] == "regression"]
        group_df = df.loc[df.index.isin(groups)].reindex(
            [group for group in groups if group in df.index]
        )
        print(
            group_df.to_string() if not group_df.empty else "No requested groups found"
        )
        missing = [group for group in groups if group not in df.index]
        if missing:
            print(f"Missing groups: {', '.join(missing)}")
        metric = "avg_Pearson_r" if "avg_Pearson_r" in group_df else "Pearson_r"
        if metric not in group_df:
            print(f"Skipping plot data: no Pearson r column in {csv_path}")
            continue
        model_label = Path(path).parts[0].removeprefix("pred_mordred_tr_")
        if model_label == "pred_mordred_avg_transformers":
            model_label = "avg_transformers"
        for group, value in pd.to_numeric(group_df[metric], errors="coerce").items():
            plot_rows.append({"model": model_label, "group": group, "Pearson_r": value})
        # mean_r2 = df["r2"].mean()
        # mean_pear = df["Pearson_r"].mean()

        # filt_df = df.loc[df.index.str.startswith("FpDensity", na=False)].copy()
        # mean_r2 = filt_df["r2"].mean()
        # mean_pear = filt_df["Pearson_r"].mean()

        # print(f"r2: {mean_r2}")
        # print(f"pear: {mean_pear}")

    plot_df = pd.DataFrame(plot_rows)
    if plot_df.empty:
        raise ValueError("No group Pearson r values found for plotting")
    plot_df.to_csv(Path(std_path) / "selected_group_pearson_by_model.csv", index=False)
    import numpy as np

    fig, ax = plt.subplots(figsize=(max(14, len(groups) * 1.4), 7))
    present_groups = [group for group in groups if group in plot_df["group"].values]
    model_order = plot_df["model"].drop_duplicates().tolist()
    import sys

    sys.path.insert(
        0, str(Path(__file__).resolve().parents[1] / "src" / "visualisation")
    )
    from vis import Visualise

    visualiser = Visualise()
    colour_aliases = {
        "chemberta-dc": "chemberta",
        "chemberta-sey": "chembertasey",
        "molformer-dc": "molformer-c3-1b",
        "molformer-ibm": "molformer",
        "avg_transformers": "selformer",
    }
    colours = {
        name: visualiser.colour_map.get(
            name, visualiser.colour_map.get(colour_aliases.get(name), "#808080")
        )
        for name in model_order
    }
    for group_i, group in enumerate(present_groups):
        group_scores = (
            plot_df.loc[plot_df["group"] == group]
            .dropna(subset=["Pearson_r"])
            .sort_values("Pearson_r", ascending=False)
        )
        n_bars = len(group_scores)
        if not n_bars:
            continue
        width = min(0.85 / n_bars, 0.16)
        offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * width
        for offset, row in zip(offsets, group_scores.itertuples()):
            ax.bar(group_i + offset, row.Pearson_r, width, color=colours[row.model])
    ax.set_xticks(range(len(present_groups)), present_groups, rotation=45, ha="right")
    ax.set_xlabel("Descriptor group")
    ax.set_ylabel("Mean Pearson r")
    ax.set_title("Mordred descriptor group prediction performance")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=colours[name], label=name) for name in model_order],
        title="Training feature set",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )
    fig.tight_layout()
    output = Path(std_path) / "selected_group_pearson_by_group.png"
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output}")


# %% MoeType performance spread without LabuteASA
if run_9:
    import numpy as np

    csv_path = Path(
        "/users/yhb18174/TL_project/results/"
        "lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/"
        "pred_mordred_avg_transformers/avg_stats_across_embedding_models.csv"
    )
    print(f"Reading {csv_path}", flush=True)
    df = pd.read_csv(csv_path, index_col=0)
    if "task_type" in df.columns:
        df = df.loc[df["task_type"] == "regression"]
    if df.index.has_duplicates:
        raise ValueError(f"Expected one row per descriptor in {csv_path}")

    # Accept common mean-column names, but never use a precomputed SD column.
    metric_columns = [
        col
        for col in ("avg_Pearson_r", "Pearson_r_mean", "mean_Pearson_r", "Pearson_r")
        if col in df.columns
    ]
    if len(metric_columns) != 1:
        raise ValueError(
            "Expected one Pearson r mean column; "
            f"found {metric_columns}. Available columns: {df.columns.tolist()}"
        )
    metric = metric_columns[0]
    # Mordred MoeType surface-area families, excluding LabuteASA.
    # Match names directly to avoid importing RDKit/Mordred and building all descriptors.
    moe_members = df.index.astype(str).str.fullmatch(
        r"(?:PEOE_VSA|SMR_VSA|SlogP_VSA|EState_VSA|VSA_EState)\d+(?:_mordred)?"
    )
    moe_df = df.loc[moe_members].copy()
    moe_df[metric] = pd.to_numeric(moe_df[metric], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    moe_df = moe_df.sort_values(metric, ascending=False, na_position="last")
    print("\nMoeType descriptors excluding LabuteASA, highest to lowest Pearson r:")
    print(moe_df.to_string())
    values = (
        pd.to_numeric(df.loc[moe_members, metric], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if len(values) < 2:
        raise ValueError(f"Need at least two finite MoeType values in {csv_path}")

    print(f"\nSource: {csv_path}\nMetric: {metric}")
    print(
        "MoeType excluding LabuteASA: sample SD across descriptor mean Pearson r values"
    )
    print(f"Descriptors: {len(values)}")
    print(f"Mean Pearson r: {values.mean():.6f}")
    print(f"SD Pearson r: {values.std(ddof=1):.6f}")
    print(f"Minimum Pearson r: {values.min():.6f}")
    print(f"Maximum Pearson r: {values.max():.6f}")
    print(f"Range Pearson r (max - min): {values.max() - values.min():.6f}")


# %% Compare MolFormer-DC, Mordred, and their combined property predictions
if run_10:
    import json
    import sys
    import numpy as np
    import matplotlib.pyplot as plt

    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "run" / "config"))
    from config import FULL_PATHING

    model_name = "random_forest"
    embedding = "molformer-dc"
    properties = [
        "aq_sol",
        "bp",
        "egfr_pic50",
        "elec_re",
        "hole_re",
        "logd",
        "log_ld50",
        "pic50",
        "pka",
        "pka_paper1_acidic",
        "pka_paper1_basic",
    ]
    labels = {
        "aq_sol": "Aqueous Solubility",
        "bp": "Boiling Point",
        "egfr_pic50": "EGFR pIC50",
        "elec_re": "Electron R.E.",
        "hole_re": "Hole R.E.",
        "logd": "logD",
        "log_ld50": "log(LD50)",
        "pic50": "pIC50",
        "pka": "pKa",
        "pka_paper1_acidic": "pKa (acidic)",
        "pka_paper1_basic": "pKa (basic)",
    }

    def score(path):
        with path.open() as handle:
            metrics = json.load(handle)["external"]["mean"]
        for key in ("pearson_r", "Pearson_r", "Pearson_R", "Pearson"):
            if key in metrics:
                return float(metrics[key])
        raise KeyError(f"No Pearson r in {path}: {list(metrics)}")

    rows = []
    for prop in properties:
        paths = FULL_PATHING["prediction_output_dirs"]["rf"].get(prop, {})
        if embedding not in paths or "mordred" not in paths:
            print(f"Skipping {prop}: feature path is not configured")
            continue
        emb_base = Path(paths[embedding]) / model_name
        files = {
            "Embedding": emb_base / f"{model_name}_performance.json",
            "Mordred": Path(paths["mordred"])
            / model_name
            / f"{model_name}_performance.json",
            "Embedding + Mordred": emb_base
            / "additional_features"
            / "mordred"
            / f"{model_name}_performance.json",
        }
        missing = [str(path) for path in files.values() if not path.is_file()]
        if missing:
            print(f"Skipping {prop}; missing: {missing}")
            continue
        rows.append((prop, {name: score(path) for name, path in files.items()}))

    if not rows:
        raise FileNotFoundError("No property has all three performance files")
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.2), 5))
    for i, name in enumerate(("Embedding", "Mordred", "Embedding + Mordred")):
        ax.bar(
            x + (i - 1) * 0.26, [scores[name] for _, scores in rows], 0.26, label=name
        )
    ax.set_xticks(x, [labels[prop] for prop, _ in rows], rotation=45, ha="right")
    ax.set_ylabel("External Pearson r")
    ax.set_xlabel("Property")
    ax.set_title("Embedding vs Mordred vs Embedding + Mordred")
    ax.legend(title="Feature Set")
    fig.tight_layout()
    output = (
        Path(FULL_PATHING["imp_dirs"]["results_dir"])
        / f"{embedding}_{model_name}_mordred_comparison.png"
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)
    print(f"Saved: {output}")


if run_11:
    import sys
    import numpy as np

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "datasets"))
    from group_descriptors import getMordredGroups

    results_root = Path("/users/yhb18174/TL_project/results")
    property_dirs = {
        "pKa": "PKA_predictions_rf",
        "Boiling Point": "BP_predictions_rf",
        "pKa basic": "PKA_P1B_predictions_rf",
        "pKa acidic": "PKA_P1A_predictions_rf",
        "pIC50 (BRD4)": "PIC50_predictions_rf",
        "log(LD50)": "LOG_LD50_predictions_rf",
        "logD": "LOGD_predictions_rf",
        "Hole R.E.": "HOLE_RE_predictions_rf",
        "Electron R.E.": "ELEC_RE_predictions_rf",
        "pIC50 (EGFR)": "EGFR_PIC50_predictions_rf",
        "Aqueous Solubility": "AQ_SOL_predictions_rf",
    }
    groups = [
        "RingCount",
        "Framework",
        "AtomCount",
        "RotatableBond",
        "TopologicalIndex",
        "Chi",
        "KappaShapeIndex",
        "WalkCount",
        "PathCount",
        "MolecularDistanceEdge",
        "GeometricalIndex",
        "PBF",
        "MoRSE",
    ]
    group_map = getMordredGroups()
    rows = []
    for prop, directory in property_dirs.items():
        files = sorted(
            (results_root / directory).glob(
                "mordred/repeats/repeat_*/*feature_importance.csv"
            )
        )
        if not files:
            print(f"Skipping {prop}: no feature-importance CSVs")
            continue
        repeat_rows = []
        for file in files:
            importance = pd.read_csv(file).set_index("Feature")["Importance"]
            importance = pd.to_numeric(importance, errors="coerce")
            if importance.index.has_duplicates:
                importance = importance.groupby(level=0).sum(min_count=1)
            for group in groups:
                members = [
                    name for name in group_map[group] if name in importance.index
                ]
                if members:
                    repeat_rows.append(
                        {
                            "property": prop,
                            "repeat": file.parent.name,
                            "group": group,
                            "importance": importance.loc[members].sum(),
                            "average_importance": importance.loc[members].mean(),
                            "n_features": len(members),
                        }
                    )
                else:
                    print(f"Missing group {group}: {prop}, {file.parent.name}")
        if not repeat_rows:
            print(f"Skipping {prop}: no requested groups matched")
            continue
        repeat_df = pd.DataFrame(repeat_rows)
        print(f"{prop}: {repeat_df['repeat'].nunique()} repeats")
        rows.append(repeat_df)

    if not rows:
        raise ValueError("No feature importance data found for the selected groups")
    all_repeats = pd.concat(rows, ignore_index=True)
    plotted_groups = [group for group in groups if group in set(all_repeats["group"])]
    print(f"Groups with at least one reported value: {', '.join(plotted_groups)}")
    all_repeats.to_csv(
        results_root / "mordred_group_importance_by_repeat.csv", index=False
    )
    summary = all_repeats.groupby(["property", "group"], as_index=False).agg(
        Importance=("importance", "mean"),
        Average_Importance=("average_importance", "mean"),
        SD=("importance", "std"),
        Average_Importance_SD=("average_importance", "std"),
        n_repeats=("importance", "count"),
        n_features=("n_features", "mean"),
    )
    summary.to_csv(
        results_root / "mordred_group_importance_by_property.csv", index=False
    )

    from matplotlib.patches import Patch

    plotted_properties = [
        prop for prop in property_dirs if prop in set(summary["property"])
    ]
    colours = dict(zip(plotted_groups, plt.get_cmap("tab20").colors))

    def plot_group_importance(value_column, ylabel, title, filename):
        fig, ax = plt.subplots(figsize=(max(18, len(plotted_properties) * 1.6), 9))
        for prop_i, prop in enumerate(plotted_properties):
            available = (
                summary.loc[
                    (summary["property"] == prop) & summary[value_column].notna(),
                    ["group", value_column],
                ]
                .set_index("group")
                .reindex(plotted_groups)
                .dropna()
            )
            if available.empty:
                continue
            width = min(0.85 / len(available), 0.16)
            offsets = (np.arange(len(available)) - (len(available) - 1) / 2) * width
            for offset, (group, row) in zip(offsets, available.iterrows()):
                ax.bar(prop_i + offset, row[value_column], width, color=colours[group])
        ax.set_xticks(
            range(len(plotted_properties)),
            plotted_properties,
            rotation=45,
            ha="right",
            fontsize=14,
        )
        ax.tick_params(axis="y", labelsize=14)
        ax.set_xlabel("Property", fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(title, fontsize=18, pad=16)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            handles=[
                Patch(color=colours[g], label=f"{g} (n={len(group_map[g])})")
                for g in plotted_groups
            ],
            title="Descriptor group",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
            fontsize=13,
            title_fontsize=14,
            frameon=False,
        )
        fig.tight_layout()
        output = results_root / filename
        fig.savefig(output, dpi=600, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output}")

    plot_group_importance(
        "Importance",
        "Mean feature importance across repeats",
        "Mordred descriptor group feature importance",
        "mordred_group_importance_by_property.png",
    )
    plot_group_importance(
        "Average_Importance",
        "Mean feature importance per descriptor across repeats",
        "Average Mordred descriptor importance by group",
        "mordred_group_average_importance_by_property.png",
    )


if run_12:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "datasets"))
    from group_descriptors import getMordredGroups

    """Select group-balanced Mordred panels from CFP performance CSVs.

    Scores are averaged over feature sets within each model, then equally over
    models. Variance is the population variance of those scores with the same
    weights (not variance of molecular values or the CSV's resampling SD).
    """

    from pathlib import Path

    import numpy as np
    import pandas as pd

    LEVELS = {
        "worst": 0.0,
        "lower_quartile": 0.25,
        "median": 0.5,
        "upper_quartile": 0.75,
        "best": 1.0,
        "low_variance": None,
        "high_variance": None,
    }

    def load_performance(sources, metric="r2"):
        """Read explicit CSV paths or glob patterns configured for each model."""
        from glob import glob

        rows, manifest = [], []
        for model, patterns in sources.items():
            paths = set()
            for pattern in patterns:
                matches = [
                    Path(p).resolve() for p in glob(str(pattern)) if Path(p).is_file()
                ]
                if not matches:
                    manifest.append(
                        dict(
                            model=model,
                            feature_set="",
                            path=str(pattern),
                            status="missing",
                            n_scores=0,
                        )
                    )
                paths.update(matches)
            seen_features = {}
            for path in sorted(paths):
                if path.suffix.lower() != ".csv" or not path.stem.startswith(
                    "pred_mordred_tr_"
                ):
                    raise ValueError(
                        f"Expected a pred_mordred_tr_<feature>.csv file: {path}"
                    )
                # Older files can use a shortened filename (e.g. molformer.csv
                # inside pred_mordred_tr_molformer-dc); prefer the experiment folder.
                experiment_name = next(
                    (
                        parent.name
                        for parent in path.parents
                        if parent.name.startswith("pred_mordred_tr_")
                    ),
                    path.stem,
                )
                train = experiment_name.removeprefix("pred_mordred_tr_")
                if train == "avg" or train.startswith("avg_"):
                    continue
                if train in seen_features:
                    raise ValueError(
                        f"Multiple CSVs for {model}/{train}: {seen_features[train]} and {path}. "
                        "Specify only the intended run in sources."
                    )
                seen_features[train] = path
                frame = pd.read_csv(path, index_col=0)
                if "model" in frame:
                    expected = "random_forest" if model == "rf_fixed_hp" else model
                    frame = frame.loc[frame["model"] == expected]
                if "task_type" in frame:
                    frame = frame.loc[frame["task_type"] == "regression"]
                if metric not in frame:
                    raise ValueError(f"Missing metric {metric!r} in {path}")
                scores = pd.to_numeric(frame[metric], errors="coerce")
                scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
                scores.index = scores.index.astype(str)
                if scores.index.has_duplicates:
                    raise ValueError(f"Duplicate descriptor rows in {path}")
                manifest.append(
                    dict(
                        model=model,
                        feature_set=train,
                        path=str(path),
                        status="loaded",
                        n_scores=len(scores),
                    )
                )
                rows.extend(
                    dict(
                        descriptor=name,
                        model=model,
                        feature_set=train,
                        score=value,
                        path=str(path),
                    )
                    for name, value in scores.items()
                )
            if not any(row["model"] == model for row in rows):
                raise ValueError(
                    f"No finite regression {metric} scores for {model}; check sources: {patterns}"
                )
        return pd.DataFrame(rows), pd.DataFrame(manifest)

    def summarise_performance(observations, group_map, models):
        """Retain coverage for every observed descriptor; require all models to select."""
        owners = {}
        for group, members in group_map.items():
            for member in members:
                if member in owners and owners[member] != group:
                    raise ValueError(f"Descriptor belongs to multiple groups: {member}")
                owners[member] = group
        data = observations.assign(score_squared=observations["score"] ** 2)
        by_model = data.groupby(["descriptor", "model"]).agg(
            mean_score=("score", "mean"),
            mean_square=("score_squared", "mean"),
            n_feature_sets=("feature_set", "nunique"),
        )
        summary = by_model.groupby("descriptor").agg(
            mean_score=("mean_score", "mean"),
            mean_square=("mean_square", "mean"),
            n_models=("mean_score", "count"),
            n_scores=("n_feature_sets", "sum"),
        )
        summary["performance_variance"] = (
            summary.pop("mean_square") - summary["mean_score"] ** 2
        ).clip(lower=0)
        summary["performance_sd"] = np.sqrt(summary["performance_variance"])
        for model in models:
            model_rows = by_model.xs(model, level="model")
            summary[f"{model}_mean"] = model_rows["mean_score"]
            summary[f"{model}_n_feature_sets"] = model_rows["n_feature_sets"]
        summary["group"] = summary.index.map(owners)
        summary["eligible"] = summary["group"].notna() & (
            summary["n_models"] == len(models)
        )
        return summary.reset_index()

    def allocate_quotas(group_map, eligible, total):
        """Proportional allocation with availability caps and deterministic rounding."""
        if total <= 0:
            raise ValueError("Selection size must be positive")
        sizes = {group: len(set(members)) for group, members in group_map.items()}
        capacity = eligible.groupby("group").size().to_dict()
        quotas = dict.fromkeys(sizes, 0)
        target = min(total, len(eligible))
        denominator = sum(sizes.values())
        ideals = {group: target * size / denominator for group, size in sizes.items()}
        for _ in range(target):
            available = [g for g in sizes if quotas[g] < capacity.get(g, 0)]
            group = min(available, key=lambda g: (-(ideals[g] - quotas[g]), g))
            quotas[group] += 1
        return quotas

    def choose_descriptor(pool, reference, category):
        if category in ("low_variance", "high_variance"):
            return pool.sort_values(
                ["performance_variance", "descriptor"],
                ascending=[category == "low_variance", True],
            ).iloc[0]
        target = reference["mean_score"].quantile(LEVELS[category])
        return (
            pool.assign(distance=(pool["mean_score"] - target).abs())
            .sort_values(["distance", "descriptor"])
            .iloc[0]
            .drop(labels="distance")
        )

    def select_panels(summary, group_map, total=250):
        """Return unique proportional set, per-group examples, and group audit.

        Examples may repeat a descriptor across criteria in small groups. The final
        set is unique; quotas below seven cannot cover all seven criteria. Larger
        quotas cycle through criteria, taking the nearest remaining descriptor.
        """
        eligible = summary.loc[summary["eligible"]].copy()
        if eligible.empty:
            raise ValueError(
                "No grouped descriptors have scores from all requested models"
            )
        quotas = allocate_quotas(group_map, eligible, total)
        selected, examples = [], []
        categories = list(LEVELS)
        for group_number, (group, reference) in enumerate(
            eligible.groupby("group", sort=True)
        ):
            for category in categories:
                row = choose_descriptor(reference, reference, category).to_dict()
                examples.append(dict(row, category=category))
            pool = reference.copy()
            # Rotate starting criteria so small groups do not all select the worst.
            for i in range(quotas[group]):
                category = categories[(i + group_number) % len(categories)]
                row = choose_descriptor(pool, reference, category)
                selected.append(dict(row, category=category))
                pool = pool.loc[pool["descriptor"] != row["descriptor"]]
        selected = pd.DataFrame(selected)
        audit = pd.DataFrame(
            {
                "group": list(group_map),
                "full_count": [len(set(v)) for v in group_map.values()],
            }
        )
        audit["eligible_count"] = (
            audit["group"].map(eligible.groupby("group").size()).fillna(0).astype(int)
        )
        audit["selected_count"] = audit["group"].map(quotas)
        audit["full_fraction"] = audit["full_count"] / audit["full_count"].sum()
        audit["selected_fraction"] = audit["selected_count"] / len(selected)
        audit["fraction_difference"] = (
            audit["selected_fraction"] - audit["full_fraction"]
        )
        return selected, pd.DataFrame(examples), audit

    def run_selection(sources, group_map, output_dir, metric="r2", total=250):
        observations, manifest = load_performance(sources, metric)
        summary = summarise_performance(observations, group_map, list(sources))
        selected, examples, audit = select_panels(summary, group_map, total)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in {
            "source_manifest": manifest,
            "descriptor_observations": observations,
            "descriptor_summary": summary,
            "selected_descriptors": selected,
            "per_group_examples": examples,
            "group_distribution": audit,
        }.items():
            frame.to_csv(output_dir / f"{name}.csv", index=False)
        (output_dir / "selected_descriptors.txt").write_text(
            "\n".join(selected["descriptor"]) + "\n", encoding="utf-8"
        )
        (output_dir / "selection_notes.txt").write_text(
            f"Metric: {metric} (higher is better)\nRequested size: {total}\n"
            f"Actual size: {len(selected)}\nModels: {', '.join(sources)}\n"
            "Mean: equal feature-set weights within models, equal model weights.\n"
            "Variance: population variance across scores with those same weights.\n"
            "Selection requires finite regression scores from every requested model.\n"
            "Feature-set coverage may differ; see descriptor_summary and source_manifest.\n"
            "Quotas target full Mordred group proportions, capped by eligible counts.\n"
            "Per-group examples can overlap; the proportional selection is unique.\n",
            encoding="utf-8",
        )
        print(
            f"Selected {len(selected)} of {summary['eligible'].sum()} eligible descriptors: {output_dir}"
        )
        if (manifest["status"] == "missing").any():
            print("Some experiments lack model CSVs; see source_manifest.csv.")
        return selected, summary, audit

    # Edit each list independently: exact CSV paths and wildcard patterns are accepted.
    # No paths.json lookup or implicit model subdirectory is added.
    # If two different files match the same model/feature set, narrow the patterns.
    sources = {
        "svr": [
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_maccs/svr/pred_mordred_tr_maccs.csv",
        ],
        "ridge": [
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/pred_mordred_tr_maccs/ridge/pred_mordred_tr_maccs.csv",
        ],
        "rf_fixed_hp": [
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_maccs/pred_mordred_tr_maccs.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_morgan/pred_mordred_tr_morgan.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_rdkit/random_forest/pred_mordred_tr_rdkit.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_molformer-dc/random_forest/pred_mordred_tr_molformer.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_molformer-ibm/random_forest/pred_mordred_tr_molformer-ibm.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_chemberta-dc/random_forest/pred_mordred_tr_chemberta-dc.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_chemberta-sey/random_forest/pred_mordred_tr_chemberta-sey.csv",
            "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_selformer/random_forest/pred_mordred_tr_selformer.csv",
        ],
    }
    output_dir = Path(
        "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/mordred_descriptor_selection"
    )
    metric = "r2"  # Higher is better; alternatively use "Pearson_r".
    # Legend label -> exact CSV path. These files are independent of selection sources.
    # Set to {} to skip plotting. Descriptor names must be in the first column.
    histogram_csvs = {
        "RDKit": "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_rdkit/random_forest/pred_mordred_tr_rdkit.csv",
        "MACCS": "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_maccs/pred_mordred_tr_maccs.csv",
        "MolFormer-DC": "/users/yhb18174/TL_project/results/lipinski_embeddings_and_descriptor_predictions/rf_fixed_hp/pred_mordred_tr_molformer-dc/random_forest/pred_mordred_tr_molformer-dc.csv",
    }
    histogram_bins = 30

    selected, summary, audit = run_selection(
        sources=sources,
        group_map=getMordredGroups(),
        output_dir=output_dir,
        metric=metric,
        total=250,  # Unique descriptors; group quotas follow full Mordred proportions.
    )

    # Optional model filters keyed by the same legend labels as histogram_csvs.
    # Otherwise use the CSV's model directory, or its sole non-null model label.
    histogram_model_filters = {}  # Example: {"MolFormer-DC": "random_forest"}

    # Compare the same CSVs before and after filtering to selected descriptors.
    if histogram_csvs:
        import matplotlib.pyplot as plt

        selected_names = set(selected["descriptor"].astype(str))
        full_distributions, selected_distributions = [], []
        for label, csv_path in histogram_csvs.items():
            frame = pd.read_csv(csv_path, index_col=0)
            if "task_type" in frame:
                frame = frame.loc[frame["task_type"] == "regression"]
            if "model" in frame:
                models = frame["model"].dropna().unique().tolist()
                chosen_model = histogram_model_filters.get(label)
                if chosen_model is None:
                    folder_model = Path(csv_path).parent.name
                    if folder_model in models:
                        chosen_model = folder_model
                    elif len(models) == 1:
                        chosen_model = models[0]
                    elif len(models) > 1:
                        raise ValueError(
                            f"Multiple models in {csv_path}: {models}. Set "
                            f"histogram_model_filters[{label!r}] to the model to plot."
                        )
                if chosen_model is not None:
                    frame = frame.loc[frame["model"] == chosen_model]
                    print(f"Histogram {label}: using model {chosen_model}")
            if metric not in frame:
                raise ValueError(f"Missing {metric!r} in histogram CSV: {csv_path}")
            scores = pd.to_numeric(frame[metric], errors="coerce")
            scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
            scores.index = scores.index.astype(str)
            if scores.index.has_duplicates:
                n_rows = len(scores)
                scores = scores.groupby(level=0, sort=False).mean()
                print(
                    f"Histogram {label}: averaged {n_rows} finite score rows into "
                    f"{len(scores)} unique descriptors in {csv_path}"
                )
            subset = scores.loc[scores.index.isin(selected_names)]
            if scores.empty or subset.empty:
                raise ValueError(
                    f"No finite scores or no selected descriptors found in histogram CSV: {csv_path}"
                )
            full_distributions.append(scores.to_numpy())
            selected_distributions.append(subset.to_numpy())

        # Shared edges across files and panels retain all finite negative R2 values.
        bin_edges = np.histogram_bin_edges(
            np.concatenate(full_distributions), bins=histogram_bins
        )
        import re

        for file_number, ((label, csv_path), full_values, selected_values) in enumerate(
            zip(histogram_csvs.items(), full_distributions, selected_distributions),
            start=1,
        ):
            fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
            count_rows = []
            for ax, panel, values, colour in zip(
                axes,
                ["All available descriptors", "Selected descriptors"],
                [full_values, selected_values],
                ["#4477AA", "#EE6677"],
            ):
                ax.hist(
                    values,
                    bins=bin_edges,
                    weights=np.full(len(values), 100.0 / len(values)),
                    color=colour,
                    edgecolor="white",
                    linewidth=0.5,
                )
                ax.set_title(f"{panel} (n={len(values)})")
                ax.set_xlabel(f"{metric} per descriptor")
                ax.spines[["top", "right"]].set_visible(False)
                counts = np.histogram(values, bins=bin_edges)[0]
                for left, right, count in zip(bin_edges[:-1], bin_edges[1:], counts):
                    count_rows.append(
                        dict(
                            panel=panel,
                            label=label,
                            source_csv=str(csv_path),
                            bin_left=left,
                            bin_right=right,
                            count=int(count),
                            percent=100.0 * count / len(values),
                            n_descriptors=len(values),
                        )
                    )
            axes[0].set_ylabel("Percentage of descriptors")
            fig.suptitle(f"{label}: full vs selected performance distributions")
            fig.tight_layout()
            safe_label = (
                re.sub(r"[^A-Za-z0-9_-]+", "_", str(label)).strip("_") or "histogram"
            )
            plot_path = (
                output_dir
                / f"{file_number:02d}_{safe_label}_{metric}_full_vs_selected_histogram.png"
            )
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            pd.DataFrame(count_rows).to_csv(plot_path.with_suffix(".csv"), index=False)
            print(f"Saved: {plot_path}")


if run_13:
    import numpy as np
    import pandas as pd
    import sys
    from pathlib import Path

    sys.path.insert(0, "/users/yhb18174/TL_project/scripts/src/visualisation")
    from vis import Visualise

    v = Visualise()

    # Load full descriptor dataset
    full_df = pd.read_csv("/users/yhb18174/TL_project/datasets/all/all_rdkit.csv")

    # Randomly select 5,000 molecules
    subset_df = full_df.sample(n=5000, random_state=42).copy()

    # Run PCA
    fig, pca_df, loadings_df, abs_loadings_df = v.plotPCA(
        data_dict={
            "Full dataset": full_df,
            "Subset": subset_df,
        },
        n_components=3,
        plot_scatter=True,
        plot_area=False,
        plot_loadings=False,
        remove_outliers=False,
        random_seed=42,
        scale=True,
        save_plot=True,
        save_path=Path("./"),
        save_fname="full_vs_subset_PCA",
        save_extra_data=True,
    )
