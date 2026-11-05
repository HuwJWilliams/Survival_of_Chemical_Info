"""
Fucntions to generate the feature importance analysis
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
import matplotlib
import re
from pathlib import Path

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import seaborn as sns

# %% ===== Project Imports & Pathing Setup =====
RUN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_DIR / "config"))

from config import PATHING_JSON_PATH, SRC_DIR, PP_ANALYSIS_METRICS

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups


def cleanFeatureName(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.index = df.index.str.replace(
        r"_(mordred|rdkit|maccs).*",
        "",
        regex=True,
    )
    return df


def AverageFeatureImportance(
    df_ls: list[str], save_path: str | Path, filename: str, top_n_feats=50
):
    if not df_ls:
        raise ValueError("No feature importance CSV files were provided.")

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    loaded_dfs = [
        pd.read_csv(df, index_col="Feature", usecols=["Importance", "Feature"])
        for df in df_ls
    ]
    mean_df = sum(loaded_dfs) / len(loaded_dfs)
    mean_df = mean_df.sort_values("Importance", ascending=False)
    plot_df = cleanFeatureName(mean_df)

    plt.figure(figsize=(10, 8))

    sns.barplot(
        data=plot_df.head(top_n_feats),
        x="Importance",
        y="Feature",
        palette="viridis",
        dodge=False,
        hue="Feature",
        legend=False,
    )

    plt.title("Feature Importances")
    plt.xlabel("Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.savefig(save_path / f"{filename}.png", dpi=800, bbox_inches="tight")
    plt.close()
    mean_df.to_csv(save_path / "feature_importance_df.csv")
    plot_df.to_csv(save_path / "feature_importance_display_df.csv")

    return mean_df


def SummativeDescriptorGroupImportance(
    feature_importance_df: pd.DataFrame,
    feature_set: str,
    save_path: str | Path,
    filename: str,
    top_n_groups: int = 50,
) -> pd.DataFrame:
    if feature_set not in {"mordred", "rdkit"}:
        raise ValueError(
            "Descriptor group importance is only supported for mordred and rdkit."
        )

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    importance_df = feature_importance_df.copy()

    if "Feature" in importance_df.columns:
        importance_df = importance_df.set_index("Feature")

    if "Importance" not in importance_df.columns:
        raise ValueError("feature_importance_df must contain an 'Importance' column.")

    group_map = getGroups(feature_set)
    group_rows = []

    importance_features = set(importance_df.index.astype(str))
    importance_base_to_feature = {}
    ambiguous_bases = set()
    for feature in importance_features:
        base_feature = re.sub(r"_(rdkit|mordred|maccs)$", "", feature)
        if base_feature in importance_base_to_feature:
            ambiguous_bases.add(base_feature)
        else:
            importance_base_to_feature[base_feature] = feature

    for group_name, group_features in group_map.items():
        present_features = []
        for feature in group_features:
            feature = str(feature)
            base_feature = re.sub(r"_(rdkit|mordred|maccs)$", "", feature)

            if feature in importance_features:
                present_features.append(feature)
            elif (
                base_feature not in ambiguous_bases
                and base_feature in importance_base_to_feature
            ):
                present_features.append(importance_base_to_feature[base_feature])

        present_features = sorted(set(present_features))

        if not present_features:
            continue

        group_rows.append(
            {
                "Group": group_name,
                "Importance": importance_df.loc[present_features, "Importance"].sum(),
                "Average_Importance": importance_df.loc[
                    present_features,
                    "Importance",
                ].mean(),
                "n_features": len(present_features),
            }
        )

    group_importance_df = pd.DataFrame(group_rows)

    if group_importance_df.empty:
        raise ValueError(
            f"No {feature_set} descriptor groups matched feature importance rows."
        )

    group_importance_df = group_importance_df.sort_values(
        "Importance",
        ascending=False,
    )

    plot_df = group_importance_df.head(top_n_groups)
    avg_plot_df = group_importance_df.sort_values(
        "Average_Importance",
        ascending=False,
    ).head(top_n_groups)

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    sns.barplot(
        data=plot_df,
        x="Importance",
        y="Group",
        hue="Group",
        palette="viridis",
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Summative Importance")
    axes[0].set_xlabel("Summed Importance")
    axes[0].set_ylabel("Descriptor Group")

    sns.barplot(
        data=avg_plot_df,
        x="Average_Importance",
        y="Group",
        hue="Group",
        palette="viridis",
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Average Importance")
    axes[1].set_xlabel("Mean Importance per Descriptor")
    axes[1].set_ylabel("Descriptor Group")

    fig.suptitle(f"{feature_set} Descriptor Group Importances", fontsize=16)
    fig.tight_layout()
    fig.savefig(save_path / f"{filename}.png", dpi=800, bbox_inches="tight")
    plt.close()

    group_importance_df.to_csv(
        save_path / "summative_descriptor_group_importance_df.csv",
        index=False,
    )

    return group_importance_df


def PlotDescriptorGroupImportanceHeatmaps(
    group_importance_df: pd.DataFrame,
    save_path: str | Path,
    value_col: str = "Average_Importance",
    dpi: int = 400,
) -> None:
    if group_importance_df.empty:
        print("No descriptor group importance rows available for heatmaps.")
        return

    required_cols = {
        "property",
        "feature_set",
        "Group",
        value_col,
        "Importance",
    }
    missing_cols = required_cols - set(group_importance_df.columns)

    if missing_cols:
        raise ValueError(
            "group_importance_df is missing columns needed for heatmaps: "
            f"{missing_cols}"
        )

    save_path = Path(save_path)
    heatmap_dir = save_path / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    def addMissingValueHatches(ax, data: pd.DataFrame) -> None:
        for row_i, (_, row) in enumerate(data.iterrows()):
            for col_i, value in enumerate(row):
                if pd.isna(value):
                    ax.add_patch(
                        Rectangle(
                            (col_i, row_i),
                            1,
                            1,
                            facecolor="white",
                            edgecolor="black",
                            hatch="//",
                            linewidth=0.5,
                        )
                    )

    viridis = colormaps.get_cmap("viridis")
    heatmap_cmap = LinearSegmentedColormap.from_list(
        "black_viridis",
        ["black", *[viridis(i / 255) for i in range(35, 256)]],
    )
    heatmap_cmap.set_bad("white")

    def saveHeatmap(
        data: pd.DataFrame,
        feature_set: str,
        filename_prefix: str,
        title: str,
        colorbar_label: str,
        annotate: bool,
    ) -> None:
        if data.empty:
            return

        figsize = (
            max(8, 0.7 * len(data.columns)),
            max(6, 0.32 * len(data)),
        )

        fig, ax = plt.subplots(figsize=figsize)

        sns.heatmap(
            data,
            annot=annotate,
            fmt=".4f",
            cmap=heatmap_cmap,
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": colorbar_label},
            ax=ax,
        )

        addMissingValueHatches(ax=ax, data=data)

        ax.set_title(
            f"{feature_set} {title}",
            fontsize=16,
            weight="bold",
        )
        ax.set_xlabel("property", fontsize=12, weight="bold")
        ax.set_ylabel("descriptor group", fontsize=12, weight="bold")
        ax.tick_params(axis="x", rotation=45)

        for label in ax.get_xticklabels():
            label.set_horizontalalignment("right")

        fig.tight_layout()

        suffix = "" if annotate else "_no_values"

        fig.savefig(
            heatmap_dir / f"{filename_prefix}{suffix}.png",
            dpi=dpi,
            bbox_inches="tight",
        )

        plt.close(fig)

    for feature_set, feature_df in group_importance_df.groupby("feature_set"):
        # Average importance
        average_df = feature_df.pivot_table(
            index="Group",
            columns="property",
            values=value_col,
            aggfunc="mean",
        )

        average_df = average_df.dropna(axis=0, how="all")

        if average_df.empty:
            print(f"No descriptor group heatmap data for {feature_set}")
            continue

        average_df = average_df.loc[
            average_df.mean(axis=1).sort_values(ascending=False).index
        ]

        average_df.to_csv(
            heatmap_dir / f"{feature_set}_descriptor_group_average_importance.csv"
        )

        # Mean raw importance
        summative_df = feature_df.pivot_table(
            index="Group",
            columns="property",
            values="Importance",
            aggfunc="mean",
        )

        summative_df = summative_df.loc[
            summative_df.mean(axis=1).sort_values(ascending=False).index
        ]

        summative_df.to_csv(
            heatmap_dir / f"{feature_set}_descriptor_group_summative_importance.csv"
        )

        # Maximum raw importance
        max_importance_df = feature_df.pivot_table(
            index="Group",
            columns="property",
            values="Importance",
            aggfunc="max",
        )

        max_importance_df = max_importance_df.loc[
            max_importance_df.mean(axis=1).sort_values(ascending=False).index
        ]

        max_importance_df.to_csv(
            heatmap_dir / f"{feature_set}_descriptor_group_max_importance.csv"
        )

        # Average heatmaps
        saveHeatmap(
            data=average_df,
            feature_set=feature_set,
            filename_prefix=(
                f"{feature_set}_descriptor_group_" "average_importance_heatmap"
            ),
            title="descriptor group average feature importance",
            colorbar_label=value_col.replace("_", " "),
            annotate=True,
        )

        saveHeatmap(
            data=average_df,
            feature_set=feature_set,
            filename_prefix=(
                f"{feature_set}_descriptor_group_" "average_importance_heatmap"
            ),
            title="descriptor group average feature importance",
            colorbar_label=value_col.replace("_", " "),
            annotate=False,
        )

        # Maximum importance heatmaps
        saveHeatmap(
            data=max_importance_df,
            feature_set=feature_set,
            filename_prefix=(
                f"{feature_set}_descriptor_group_" "max_importance_heatmap"
            ),
            title="descriptor group maximum feature importance",
            colorbar_label="Maximum importance",
            annotate=True,
        )

        saveHeatmap(
            data=max_importance_df,
            feature_set=feature_set,
            filename_prefix=(
                f"{feature_set}_descriptor_group_" "max_importance_heatmap"
            ),
            title="descriptor group maximum feature importance",
            colorbar_label="Maximum importance",
            annotate=False,
        )


def getFeatureImportanceFiles(feature_path: Path) -> list[Path]:
    return sorted((feature_path / "repeats").glob("repeat_*/*_feature_importance.csv"))


def getFeatureRunLabel(feature_path: Path, feature_set: str) -> str:
    return feature_set


def runFeatureImportanceAnalysis(
    properties: list[str],
    feature_sets: list[str],
    preds_dir: dict,
    save_dir: str | Path,
    top_n_feats: int = 50,
) -> tuple[list[tuple], list[tuple]]:
    save_root = Path(save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    failed_runs = []
    completed_runs = []
    descriptor_group_importance_rows = []

    for prop in properties:
        if prop not in preds_dir:
            print(f"{prop} not in prediction output paths")
            continue

        print(f"Processing {prop} feature importances...")

        for feature_set in feature_sets:
            if feature_set not in preds_dir[prop]:
                print(f"{feature_set} not available for {prop}")
                continue

            feature_path = Path(preds_dir[prop][feature_set])
            feature_importance_files = getFeatureImportanceFiles(feature_path)

            if not feature_importance_files:
                print(f"No feature importance files found for {prop} / {feature_path}")
                failed_runs.append((prop, str(feature_path), "no files"))
                continue

            run_label = getFeatureRunLabel(
                feature_path=feature_path,
                feature_set=feature_set,
            )
            run_save_path = save_root / prop / run_label
            filename = f"{prop}_{run_label}_average_feature_importance"

            try:
                avg_feature_importance_df = AverageFeatureImportance(
                    df_ls=[str(path) for path in feature_importance_files],
                    save_path=run_save_path,
                    filename=filename,
                    top_n_feats=top_n_feats,
                )

                if feature_set in {"mordred", "rdkit"}:
                    descriptor_group_importance_df = SummativeDescriptorGroupImportance(
                        feature_importance_df=avg_feature_importance_df,
                        feature_set=feature_set,
                        save_path=run_save_path,
                        filename=f"{prop}_{run_label}_summative_descriptor_group_importance",
                    )
                    descriptor_group_importance_df["property"] = prop
                    descriptor_group_importance_df["feature_set"] = feature_set
                    descriptor_group_importance_df["run_label"] = run_label
                    descriptor_group_importance_rows.append(
                        descriptor_group_importance_df
                    )

                completed_runs.append((prop, run_label, len(feature_importance_files)))
                print(
                    f"Saved average feature importance for {prop} / {run_label} "
                    f"from {len(feature_importance_files)} files"
                )
            except Exception as e:
                failed_runs.append((prop, str(feature_path), str(e)))
                print(f"Failed {prop} / {feature_path}:\n{e}")

    print(f"Completed feature-importance runs: {len(completed_runs)}")

    if failed_runs:
        print("Failed or missing feature-importance runs:")
        for prop, feature_path, reason in failed_runs:
            print(f"  {prop} / {feature_path}: {reason}")

    if descriptor_group_importance_rows:
        descriptor_group_importance_df = pd.concat(
            descriptor_group_importance_rows,
            ignore_index=True,
        )
        descriptor_group_importance_df = descriptor_group_importance_df[
            [
                "property",
                "feature_set",
                "run_label",
                "Group",
                "Importance",
                "Average_Importance",
                "n_features",
            ]
        ]
        descriptor_group_importance_df.to_csv(
            save_root / "descriptor_group_importance.csv",
            index=False,
        )
        PlotDescriptorGroupImportanceHeatmaps(
            group_importance_df=descriptor_group_importance_df,
            save_path=save_root,
            value_col="Average_Importance",
        )

    return completed_runs, failed_runs


def importanceVsPredictability(
    feature_importance_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    performance_metric: str = "r2",
):
    feat_imp_df = cleanFeatureName(feature_importance_df)

    pred_df = pred_df.copy()
    if "Feature" in pred_df.columns:
        pred_df = pred_df.set_index("Feature")

    pred_df = cleanFeatureName(pred_df)

    plot_df = pd.DataFrame(
        {
            "importance": feat_imp_df,
            performance_metric: pred_df[performance_metric],
        }
    ).dropna()

    fig, ax1 = plt.subplots(figsize=(12, 5))

    x = range(len(plot_df))

    ax1.scatter(x, plot_df["importance"], color="tab:blue", label="Importance")
    ax1.set_ylabel("Importance", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.scatter(
        x,
        plot_df[performance_metric],
        color="tab:orange",
        label=f"Predictability ({performance_metric})",
    )
    ax2.set_ylabel(f"Predictability ({performance_metric})", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_xticks(x)
    ax1.set_xticklabels(plot_df.index, rotation=90)
    ax1.set_xlabel("Descriptor")

    ax1.set_title("Feature Importance vs Descriptor Predictability")

    fig.tight_layout()
    plt.show()

    return plot_df
