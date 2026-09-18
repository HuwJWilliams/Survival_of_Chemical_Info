"""
Functions to analyse the individual property prediction (PP) results
"""

# %% ===== Python Imports =====
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
from glob import glob
import numpy as np

# %% ===== Project Imports & Pathing Setup =====
RUN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_DIR / "config"))

from config import PATHING_JSON_PATH, SRC_DIR

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import checkLipinskiCriteria

v = Visualise(save_all=False)


# %% ===== Function Definitions =====
def selectPPAnalysisInputs(preds_dir, target_paths, target_columns, properties, feature_sets=None):
    """Use configured feature names and exclude properties without target metadata."""
    valid_properties = []
    for prop in properties:
        if prop not in target_columns or prop not in target_paths:
            print(f"Skipping {prop}: missing target column or target path configuration")
            continue
        valid_properties.append(prop)
    if feature_sets is None:
        feature_sets = list(dict.fromkeys(
            feature for prop in valid_properties for feature in preds_dir[prop]
        ))
    return valid_properties, [feature for feature in feature_sets if "-random-" not in feature]


def getPropertyPerformanceDfs(
    property_pathing: dict[str, Path],
    feature_sets: list[str],
    perf_fname: str = "rf_performance.json",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read internal and external mean/std performance from rf_performance.json
    files for each feature set.

    Returns
    -------
    int_property_performance_df:
        Rows for internal mean/std performance.

    ext_property_performance_df:
        Rows for external mean/std performance.
    """

    int_rows = []
    ext_rows = []

    for feat in feature_sets:
        if feat not in property_pathing:
            continue
        feature_path = Path(property_pathing[feat])
        perf_json_path = feature_path / perf_fname

        try:
            with open(perf_json_path, "r") as f:
                perf_json = json.load(f)
            internal = perf_json["internal"]
            int_mean = internal["mean"]
            int_std = internal["std"]
            external = perf_json["external"]
            ext_mean = external["mean"]
            ext_std = external["std"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Skipping performance for {feat} ({perf_json_path}): {exc}")
            continue

        int_rows.append(
            {
                "feature_set": feat,
                "stat": "mean",
                **int_mean,
            }
        )

        int_rows.append(
            {
                "feature_set": feat,
                "stat": "std",
                **int_std,
            }
        )

        ext_rows.append(
            {
                "feature_set": feat,
                "stat": "mean",
                **ext_mean,
            }
        )

        ext_rows.append(
            {
                "feature_set": feat,
                "stat": "std",
                **ext_std,
            }
        )

    int_property_performance_df = pd.DataFrame(int_rows)
    ext_property_performance_df = pd.DataFrame(ext_rows)

    return int_property_performance_df, ext_property_performance_df


def getFeatureColourConfig(
    feature_ls: list[str],
    colour_map: dict[str, str],
):
    feature_colour_config = {}

    for feature in feature_ls:
        if feature.startswith("ft-"):
            hatch = "//"
            stripped_feature_name = feature.removeprefix("ft-")
            stripped_feature_name = stripped_feature_name.removeprefix("random-")
            stripped_feature_name = stripped_feature_name.removeprefix("scaffold-")
        else:
            hatch = None
            stripped_feature_name = feature

        colour_aliases = {
            "chemberta-dc": "chemberta", "chemberta-sey": "chembertasey",
            "molformer-ibm": "molformer", "molformer-dc": "molformer-c3-1b",
        }
        colour = colour_map.get(
            stripped_feature_name,
            colour_map.get(colour_aliases.get(stripped_feature_name), "#808080"),
        )
        feature_colour_config[feature] = (colour, hatch)

    return feature_colour_config


FT_FEATURE_PAIRS = {
    **{
        f"ft-{split}-{base}": base
        for split in ("scaffold", "random")
        for base in ("chemberta-dc", "chemberta-sey", "molformer-ibm", "molformer-dc")
    },
    "ft-scaffold-chemberta": "chemberta",
    "ft-scaffold-chembertasey": "chembertasey",
    "ft-scaffold-molformer": "molformer",
    "ft-scaffold-molformer-c3-1b": "molformer-c3-1b",
    "ft-scaffold-selformer": "selformer",
    "ft-random-chemberta": "chemberta",
    "ft-random-chembertasey": "chembertasey",
    "ft-random-molformer": "molformer",
    "ft-random-molformer-c3-1b": "molformer-c3-1b",
    "ft-random-selformer": "selformer",
}

METRIC_ALIASES = {
    "pearson_r": ["pearson_r", "r_pearson", "Pearson_r"],
    "Pearson_r": ["Pearson_r", "pearson_r", "r_pearson"],
    "r_pearson": ["r_pearson", "pearson_r", "Pearson_r"],
    "rmse": ["rmse", "RMSE"],
    "RMSE": ["RMSE", "rmse"],
    "bias": ["bias", "Bias"],
    "Bias": ["Bias", "bias"],
    "sdep": ["sdep", "SDEP"],
    "SDEP": ["SDEP", "sdep"],
}


def getMetricColumn(metric: str, data: pd.DataFrame) -> str | None:
    for metric_alias in METRIC_ALIASES.get(metric, []):
        if metric_alias in data.columns:
            return metric_alias

    metric_lower = metric.lower()

    if metric in data.columns:
        return metric

    if metric_lower in data.columns:
        return metric_lower

    return None


def getFeatureStyleFromConfig(
    colour_map: dict | None,
    feature: str,
    fallback_feature: str | None = None,
    default: str = "#808080",
) -> tuple[str, str | None]:
    if colour_map is None:
        return default, "//" if feature.startswith("ft-") else None

    style = colour_map.get(feature)

    if style is None and fallback_feature is not None:
        style = colour_map.get(fallback_feature)

    if style is None:
        return default, "//" if feature.startswith("ft-") else None

    if isinstance(style, tuple):
        colour, hatch = style
        return colour, hatch

    if isinstance(style, dict):
        colour = style.get("colour", default)
        hatch = style.get("hatch")
        return colour, hatch

    return style, "//" if feature.startswith("ft-") else None


def getMatchedFTPerformanceDf(
    prop, feature_sets, full_pathing, preds_dir, target_columns, save_path,
) -> pd.DataFrame:
    """Score each FT/base pair on identical finite external observations.

    Use the saved mean out-of-sample prediction per molecule for both models.
    Save the aligned inputs as an audit trail for the comparison scores.
    """
    target_col = target_columns[prop]
    target_df = pd.read_csv(full_pathing["targets"][prop], dtype={"ID": str})
    true = pd.to_numeric(target_df.set_index("ID")[target_col], errors="coerce")
    true = true.replace([np.inf, -np.inf], np.nan).dropna().rename("true")
    if not true.index.is_unique:
        raise ValueError(f"Duplicate target molecule IDs for {prop}")
    q1, q3 = true.quantile([0.25, 0.75])
    lower, upper = q1 - 3 * (q3 - q1), q3 + 3 * (q3 - q1)
    predictions = {}
    rows = []
    for ft_feature, base_feature in FT_FEATURE_PAIRS.items():
        if not {ft_feature, base_feature}.issubset(feature_sets):
            continue
        for feature in (ft_feature, base_feature):
            if feature in predictions:
                continue
            directory = preds_dir.get(prop, {}).get(feature)
            path = Path(directory) / "last_20pct_pred.csv.gz" if directory else None
            if path is None or not path.exists():
                print(f"Skipping matched comparison: missing predictions for {prop}/{feature}")
                predictions[feature] = None
                continue
            frame = pd.read_csv(path, dtype={"ID": str}).set_index("ID")
            if not frame.index.is_unique:
                raise ValueError(f"Duplicate prediction molecule IDs in {path}")
            column = target_col if target_col in frame else frame.columns[0]
            predictions[feature] = pd.to_numeric(frame[column], errors="coerce")
        if any(predictions[feature] is None for feature in (ft_feature, base_feature)):
            continue
        matched = pd.concat(
            [true, predictions[ft_feature].rename("ft_pred"),
             predictions[base_feature].rename("base_pred")],
            axis=1, join="inner",
        ).replace([np.inf, -np.inf], np.nan).dropna()
        for split, frame in (
            ("external", matched),
            ("external_3xIQR", matched.loc[matched["true"].between(lower, upper)]),
        ):
            if len(frame) < 2:
                print(f"Skipping {prop}/{split}/{ft_feature}: fewer than two shared molecules")
                continue
            directory = Path(save_path) / prop / split / "ft_differences"
            directory.mkdir(parents=True, exist_ok=True)
            frame.to_csv(directory / f"{ft_feature}_vs_{base_feature}_matched_predictions.csv", index_label="ID")
            for feature, column in ((ft_feature, "ft_pred"), (base_feature, "base_pred")):
                error = frame[column] - frame["true"]
                rows.append({
                    "property": prop, "split": split, "feature_set": feature,
                    "comparison_pair": ft_feature, "stat": "mean", "n": len(frame),
                    "pearson_r": frame["true"].corr(frame[column]),
                    "r2": calculateR2(frame["true"], frame[column]),
                    "rmse": error.pow(2).mean() ** 0.5,
                    "bias": error.mean(),
                    "sdep": (error - error.mean()).pow(2).mean() ** 0.5,
                })
    return pd.DataFrame(rows)


def getFTDifferenceDf(
    data: pd.DataFrame,
    prop: str,
    split_name: str,
    metric_col: str,
) -> pd.DataFrame:
    if data.empty:
        return pd.DataFrame()
    if "comparison_pair" not in data or "n" not in data:
        raise ValueError("FT differences require scores calculated on matched molecule IDs")
    mean_df = data.loc[
        data["stat"] == "mean",
        ["feature_set", "comparison_pair", "n", metric_col],
    ].copy()

    mean_df[metric_col] = pd.to_numeric(mean_df[metric_col], errors="coerce")

    rows = []
    for ft_feature, base_feature in FT_FEATURE_PAIRS.items():
        pair_df = mean_df.loc[mean_df["comparison_pair"] == ft_feature].set_index("feature_set")
        mean_by_feature = pair_df[metric_col]
        if (
            ft_feature not in mean_by_feature.index
            or base_feature not in mean_by_feature.index
        ):
            continue

        rows.append(
            {
                "property": prop,
                "split": split_name,
                "metric": metric_col,
                "feature_pair": f"{ft_feature} - {base_feature}",
                "ft_feature": ft_feature,
                "base_feature": base_feature,
                "ft_value": mean_by_feature[ft_feature],
                "base_value": mean_by_feature[base_feature],
                "n_matched": int(pair_df.loc[ft_feature, "n"]),
                "difference": mean_by_feature[ft_feature]
                - mean_by_feature[base_feature],
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "property",
                "split",
                "metric",
                "feature_pair",
                "ft_feature",
                "base_feature",
                "ft_value",
                "base_value",
                "difference",
            ]
        )

    return pd.DataFrame(rows).dropna(subset=["difference"])


def plotFTDifferenceBar(
    data: pd.DataFrame,
    prop: str,
    split_name: str,
    metric_col: str,
    save_path: Path,
    colour_map: dict,
    dpi: int = 400,
) -> pd.DataFrame:
    diff_df = getFTDifferenceDf(
        data=data,
        prop=prop,
        split_name=split_name,
        metric_col=metric_col,
    )

    if diff_df.empty:
        print(f"No FT/base pairs for {prop} / {split_name} / {metric_col}")
        return diff_df

    diff_df = diff_df.sort_values("difference", ascending=False)
    feature_pair_order = diff_df["feature_pair"].tolist()
    feature_pair_styles = {
        row["feature_pair"]: getFeatureStyleFromConfig(
            colour_map=colour_map,
            feature=row["ft_feature"],
            fallback_feature=row["base_feature"],
        )
        for _, row in diff_df.iterrows()
    }
    feature_pair_palette = {
        feature_pair: feature_pair_styles[feature_pair][0]
        for feature_pair in feature_pair_order
    }

    plot_dir = save_path / prop / split_name / "ft_differences"
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=diff_df,
        x="feature_pair",
        y="difference",
        hue="feature_pair",
        order=feature_pair_order,
        hue_order=feature_pair_order,
        palette=feature_pair_palette,
        legend=False,
    )
    for container, feature_pair in zip(ax.containers, feature_pair_order):
        hatch = feature_pair_styles[feature_pair][1]
        if hatch:
            for bar in container:
                bar.set_hatch(hatch)
                bar.set_edgecolor("black")
                bar.set_linewidth(0.8)

    ax.axhline(0, color="black", linewidth=1)
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    plt.yticks(fontsize=10)
    plt.ylabel(f"ft - base {metric_col}", fontsize=12, weight="bold")
    plt.xlabel("feature pair", fontsize=12, weight="bold")
    plt.title(
        f"{prop}: {split_name} {metric_col} FT difference (shared molecules)",
        fontsize=16,
        weight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        plot_dir / f"{prop}_{split_name}_{metric_col}_ft_difference_bar.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()

    return diff_df


def plotSummaryHeatmaps(
    summary_df: pd.DataFrame,
    split_name: str,
    save_path: Path,
    metrics: list[str],
    feature_order: list[str],
    dpi: int = 400,
) -> None:
    heatmap_dir = save_path / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)

    lower_is_better_metrics = {"rmse", "mse", "mae", "sdep"}

    for metric in metrics:
        if metric not in summary_df.columns:
            print(f"{metric} not in {split_name} summary columns")
            continue

        heatmap_df = summary_df.pivot(
            index="property",
            columns="feature_set",
            values=metric,
        )

        ordered_features = [
            feature for feature in feature_order if feature in heatmap_df.columns
        ]
        heatmap_df = heatmap_df[ordered_features]

        if heatmap_df.empty:
            print(f"No heatmap data for {split_name} / {metric}")
            continue

        figsize = (14, max(6, 0.45 * len(heatmap_df)))

        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt=".3f",
            cmap="viridis",
            linewidths=0.5,
            linecolor="white",
        )
        plt.title(
            f"{split_name} average {metric}",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("feature_set", fontsize=12, weight="bold")
        plt.ylabel("property", fontsize=12, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            heatmap_dir / f"{split_name}_average_{metric}_heatmap.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=figsize)
        sns.heatmap(
            heatmap_df,
            annot=False,
            cmap="viridis",
            linewidths=0.5,
            linecolor="white",
        )
        plt.title(
            f"{split_name} average {metric}",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("feature_set", fontsize=12, weight="bold")
        plt.ylabel("property", fontsize=12, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            heatmap_dir / f"{split_name}_average_{metric}_heatmap_no_values.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()

        if metric.lower() == "bias":
            ranking_df = heatmap_df.abs().rank(
                axis=1,
                method="min",
                ascending=True,
            )
        else:
            ranking_df = heatmap_df.rank(
                axis=1,
                method="min",
                ascending=metric.lower() in lower_is_better_metrics,
            )

        plt.figure(figsize=figsize)
        sns.heatmap(
            ranking_df,
            annot=True,
            fmt=".0f",
            cmap="RdYlGn_r",
            linewidths=0.5,
            linecolor="white",
            cbar_kws={"label": "placement rank (1 = best)"},
        )
        plt.title(
            f"{split_name} average {metric} placement",
            fontsize=16,
            weight="bold",
        )
        plt.xlabel("feature_set", fontsize=12, weight="bold")
        plt.ylabel("property", fontsize=12, weight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            heatmap_dir / f"{split_name}_average_{metric}_placement_heatmap.png",
            dpi=dpi,
            bbox_inches="tight",
        )
        plt.close()


def plotGroupedPropertyFeatureBar(
    summary_df: pd.DataFrame,
    split_name: str,
    save_path: Path,
    metric: str = "pearson_r",
    feature_order: list[str] | None = None,
    colour_map: dict | None = None,
    dpi: int = 400,
) -> None:
    if metric not in summary_df.columns:
        print(f"{metric} not in {split_name} summary columns")
        return

    plot_df = summary_df[["property", "feature_set", metric]].copy()
    plot_df[metric] = pd.to_numeric(plot_df[metric], errors="coerce")
    plot_df = plot_df.dropna(subset=[metric])

    if plot_df.empty:
        print(f"No grouped bar data for {split_name} / {metric}")
        return

    if feature_order is None:
        feature_order = plot_df["feature_set"].drop_duplicates().tolist()

    feature_order = [
        feature
        for feature in feature_order
        if feature in plot_df["feature_set"].unique()
    ]
    property_order = plot_df["property"].drop_duplicates().tolist()

    if colour_map is None:
        colour_map = {}

    palette = {}
    hatches = {}
    for feature in feature_order:
        style = colour_map.get(feature, "#808080")
        if isinstance(style, tuple):
            colour, hatch = style
        elif isinstance(style, dict):
            colour = style.get("colour", "#808080")
            hatch = style.get("hatch")
        else:
            colour = style
            hatch = "//" if feature.startswith("ft-") else None

        palette[feature] = colour
        hatches[feature] = hatch

    bar_dir = save_path / "grouped_bars"
    bar_dir.mkdir(parents=True, exist_ok=True)

    figsize = (max(14, 0.85 * len(property_order)), 7)
    plt.figure(figsize=figsize)
    ax = sns.barplot(
        data=plot_df,
        x="property",
        y=metric,
        hue="feature_set",
        order=property_order,
        hue_order=feature_order,
        palette=palette,
        errorbar=None,
    )

    for container, feature in zip(ax.containers, feature_order):
        hatch = hatches.get(feature)
        if hatch:
            for bar in container:
                bar.set_hatch(hatch)
                bar.set_edgecolor("black")
                bar.set_linewidth(0.8)

    if metric.lower() == "r2":
        ax.set_ylim(0, 1)

    ax.set_xlabel("property", fontsize=12, weight="bold")
    ax.set_ylabel(metric, fontsize=12, weight="bold")
    ax.set_title(
        f"{split_name} average {metric} by property and feature set",
        fontsize=16,
        weight="bold",
    )
    ax.tick_params(axis="x", labelrotation=45, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.legend(
        title="feature_set",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    plt.tight_layout()
    plt.savefig(
        bar_dir / f"{split_name}_average_{metric}_grouped_property_feature_bar.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close()


def plotFTDifferenceSummaryBars(
    ft_difference_df: pd.DataFrame,
    save_path: Path,
    metrics: list[str] | None = None,
    colour_map: dict | None = None,
    dpi: int = 400,
) -> None:
    if ft_difference_df.empty:
        print("No FT difference rows available for summary bar plots.")
        return

    if metrics is None:
        metrics = ["pearson_r"]

    plot_dir = save_path / "ft_differences"
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_alias_lookup = {
        metric: METRIC_ALIASES.get(metric, []) + [metric] for metric in metrics
    }

    for split_name in sorted(ft_difference_df["split"].dropna().unique()):
        split_df = ft_difference_df.loc[ft_difference_df["split"] == split_name].copy()

        for metric, aliases in metric_alias_lookup.items():
            metric_df = split_df.loc[split_df["metric"].isin(aliases)].copy()
            metric_df["difference"] = pd.to_numeric(
                metric_df["difference"],
                errors="coerce",
            )
            metric_df = metric_df.dropna(subset=["difference"])

            if metric_df.empty:
                print(f"No FT difference summary rows for {split_name} / {metric}")
                continue

            property_order = metric_df["property"].drop_duplicates().tolist()
            feature_pair_order = metric_df["feature_pair"].drop_duplicates().tolist()
            feature_pair_styles = {
                row["feature_pair"]: getFeatureStyleFromConfig(
                    colour_map=colour_map,
                    feature=row["ft_feature"],
                    fallback_feature=row["base_feature"],
                )
                for _, row in metric_df.drop_duplicates("feature_pair").iterrows()
            }
            feature_pair_palette = {
                feature_pair: feature_pair_styles[feature_pair][0]
                for feature_pair in feature_pair_order
            }

            plt.figure(figsize=(max(14, 0.9 * len(property_order)), 7))
            ax = sns.barplot(
                data=metric_df,
                x="property",
                y="difference",
                hue="feature_pair",
                order=property_order,
                hue_order=feature_pair_order,
                palette=feature_pair_palette,
                errorbar=None,
            )
            for container, feature_pair in zip(ax.containers, feature_pair_order):
                hatch = feature_pair_styles[feature_pair][1]
                if hatch:
                    for bar in container:
                        bar.set_hatch(hatch)
                        bar.set_edgecolor("black")
                        bar.set_linewidth(0.8)

            ax.axhline(0, color="black", linewidth=1)
            property_labels = {
                "bp": "Boiling Point", "logd": "logD", "pka": "pKa (OChem)",
                "pka_paper1_basic": "pKa (basic)", "pka_paper1_acidic": "pKa (acidic)",
                "log_ld50": "log(LD50)", "pic50": "BRD4 pIC50",
                "hole_re": "Hole R.E", "elec_re": "Electron R.E",
                "aq_sol": "Aqueous Solubility", "egfr_pic50": "EGFR pIC50",
            }
            ax.set_xticks(range(len(property_order)))
            ax.set_xticklabels(
                [property_labels.get(prop, prop) for prop in property_order],
                rotation=45, ha="right", fontweight="bold",
            )
            ax.set_xlabel("Property", fontsize=12, weight="bold")
            metric_label = "Pearson r" if metric in METRIC_ALIASES["pearson_r"] else metric
            ax.set_ylabel(
                f"Δ {metric_label}\n(Fine-Tuned vs. Pretrained)", fontsize=12, weight="bold",
            )
            ax.set_title(
                "Fine-Tuned vs Pretrained Embeddings\n"
                f"{split_name.replace('_', ' ')} · shared molecules",
                fontsize=16,
                weight="bold",
            )
            ax.tick_params(axis="x", labelrotation=45, labelsize=10)
            ax.tick_params(axis="y", labelsize=10)
            feature_labels = {
                "selformer": "SELFormer", "molformer": "MolFormer-IBM",
                "molformer-ibm": "MolFormer-IBM", "molformer-c3-1b": "MolFormer-DC",
                "molformer-dc": "MolFormer-DC", "chemberta": "ChemBERTa-DC",
                "chemberta-dc": "ChemBERTa-DC", "chembertasey": "ChemBERTa-Sey",
                "chemberta-sey": "ChemBERTa-Sey",
            }
            pair_bases = metric_df.drop_duplicates("feature_pair").set_index("feature_pair")["base_feature"]
            handles, labels = ax.get_legend_handles_labels()
            display_labels = [feature_labels.get(pair_bases[label], pair_bases[label]) for label in labels]
            # Keep distinct pairs identifiable if aliases or multiple FT variants coexist.
            display_labels = [
                f"{display} ({label})" if display_labels.count(display) > 1 else display
                for label, display in zip(labels, display_labels)
            ]
            ax.legend(
                handles, display_labels,
                title="Feature Set",
                bbox_to_anchor=(1.02, 1),
                loc="upper left",
                borderaxespad=0,
            )
            plt.tight_layout()
            plt.savefig(
                plot_dir / f"{split_name}_{metric}_ft_difference_summary_bar.png",
                dpi=dpi,
                bbox_inches="tight",
            )
            plt.close()
            print(f"Saved FT comparison summary: {plot_dir / f'{split_name}_{metric}_ft_difference_summary_bar.png'}")


def loadFeaturePattern(path: str | Path) -> pd.DataFrame:
    path = Path(path) if "*" not in str(path) else str(path)

    if "*" in str(path):
        files = sorted(glob(str(path)))
        if not files:
            raise FileNotFoundError(f"No files matched: {path}")
        return pd.concat(
            [pd.read_csv(file, index_col=0, low_memory=False) for file in files],
            axis=0,
        )

    return pd.read_csv(path, index_col=0, low_memory=False)


def calculateR2(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_pred = pd.to_numeric(y_pred, errors="coerce")
    valid = y_true.notna() & y_pred.notna()
    y_true = y_true.loc[valid]
    y_pred = y_pred.loc[valid]

    if len(y_true) < 2:
        return float("nan")

    ss_total = ((y_true - y_true.mean()) ** 2).sum()
    if ss_total == 0:
        return float("nan")

    ss_res = ((y_true - y_pred) ** 2).sum()
    return float(1 - (ss_res / ss_total))


def plotTrueVsPred(
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    true_col: str,
    pred_col: str,
    save_path: str | Path,
    save_fname: str = "true_vs_pred_scatter",
    model_name: str | None = None,
    dpi: int = 300,
    remove_pred_outliers: bool = False,
    remove_true_outliers: bool = False,
    lower_pct: float = 1,
    upper_pct: float = 99,
) -> dict:
    true_df = true_df.copy()
    pred_df = pred_df.copy()

    true_df.index = true_df.index.astype(str)
    pred_df.index = pred_df.index.astype(str)

    common_idx = true_df.index.intersection(pred_df.index)

    y_true = pd.to_numeric(true_df.loc[common_idx, true_col], errors="coerce")
    y_pred = pd.to_numeric(pred_df.loc[common_idx, pred_col], errors="coerce")

    plot_df = pd.DataFrame({"true": y_true, "pred": y_pred}).dropna()

    if plot_df.empty:
        print(f"No valid true/pred rows for {model_name or save_fname}")
        return {}

    if remove_pred_outliers:
        pred_low = plot_df["pred"].quantile(lower_pct / 100)
        pred_high = plot_df["pred"].quantile(upper_pct / 100)
        plot_df = plot_df[plot_df["pred"].between(pred_low, pred_high)]

    if remove_true_outliers:
        true_low = plot_df["true"].quantile(lower_pct / 100)
        true_high = plot_df["true"].quantile(upper_pct / 100)
        plot_df = plot_df[plot_df["true"].between(true_low, true_high)]

    if len(plot_df) < 2:
        print(f"Not enough rows to plot true vs pred for {model_name or save_fname}")
        return {}

    y_true = plot_df["true"]
    y_pred = plot_df["pred"]

    residuals = y_pred - y_true
    bias = residuals.mean()
    sdep = ((residuals - bias).pow(2).mean()) ** 0.5
    rmse = (residuals.pow(2).mean()) ** 0.5
    r2 = calculateR2(y_true, y_pred)
    pearson_r = y_true.corr(y_pred, method="pearson")

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "pearson_r": float(pearson_r),
        "bias": float(bias),
        "sdep": float(sdep),
        "n": int(len(plot_df)),
    }

    fig, ax = plt.subplots(figsize=(6, 7))

    ax.scatter(
        y_true,
        y_pred,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.4,
    )

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())

    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--",
        color="red",
        linewidth=1,
    )

    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(f"True vs Predicted ({model_name or save_fname})")

    table_data = [[f"{pearson_r:.3f}"]]

    table = ax.table(
        cellText=table_data,
        colLabels=["Pearson r"],
        cellLoc="center",
        loc="bottom",
        bbox=[0.0, -0.32, 1.0, 0.16],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path / f"{save_fname}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return metrics


def plotTrainTestPredDistribution(
    train_df: pd.DataFrame,
    true_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    train_col: str,
    true_col: str,
    pred_col: str,
    save_path: str | Path,
    save_fname: str = "train_test_pred_distribution",
    bins: int = 40,
    alpha: float = 0.45,
    density: bool = False,
    dpi: int = 300,
) -> dict:
    train_df = train_df.copy()
    true_df = true_df.copy()
    pred_df = pred_df.copy()

    train_df.index = train_df.index.astype(str)
    true_df.index = true_df.index.astype(str)
    pred_df.index = pred_df.index.astype(str)

    common_idx = true_df.index.intersection(pred_df.index)

    train_vals = pd.to_numeric(train_df[train_col], errors="coerce").dropna()
    test_vals = pd.to_numeric(
        true_df.loc[common_idx, true_col], errors="coerce"
    ).dropna()
    pred_vals = pd.to_numeric(
        pred_df.loc[common_idx, pred_col], errors="coerce"
    ).dropna()

    if train_vals.empty or test_vals.empty or pred_vals.empty:
        print(
            f"Skipping distribution plot for {save_fname}: missing train/test/pred values"
        )
        return {}

    all_vals = pd.concat([train_vals, test_vals, pred_vals])

    if all_vals.nunique() < 2:
        print(f"Skipping distribution plot for {save_fname}: not enough value range")
        return {}

    bin_edges = np.linspace(all_vals.min(), all_vals.max(), bins + 1)

    fig, ax = plt.subplots(figsize=(11, 6))

    ax.hist(
        train_vals,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"Train true (n={len(train_vals)})",
        edgecolor="black",
        color="tab:blue",
    )
    ax.hist(
        test_vals,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"Test true (n={len(test_vals)})",
        edgecolor="black",
        color="tab:orange",
    )
    ax.hist(
        pred_vals,
        bins=bin_edges,
        alpha=alpha,
        density=density,
        label=f"Predicted (n={len(pred_vals)})",
        edgecolor="black",
        color="tab:green",
    )

    ax.set_xlabel(true_col)
    ax.set_ylabel("Density" if density else "Count")
    ax.set_title(save_fname.replace("_", " "))
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        save_path / f"{save_fname}.png",
        dpi=dpi,
        bbox_inches="tight",
    )
    plt.close(fig)

    return {
        "n_train": int(len(train_vals)),
        "n_test": int(len(test_vals)),
        "n_pred": int(len(pred_vals)),
        "train_min": float(train_vals.min()),
        "train_max": float(train_vals.max()),
        "test_min": float(test_vals.min()),
        "test_max": float(test_vals.max()),
        "pred_min": float(pred_vals.min()),
        "pred_max": float(pred_vals.max()),
        "train_mean": float(train_vals.mean()),
        "test_mean": float(test_vals.mean()),
        "pred_mean": float(pred_vals.mean()),
    }


def plotTarget3xIQRDistribution(
    target_df: pd.DataFrame,
    target_col: str,
    save_path: str | Path,
    save_fname: str = "target_distribution_3xIQR",
    iqr_multiplier: float = 3.0,
    bins: int = 40,
    dpi: int = 300,
) -> dict:
    values = pd.to_numeric(target_df[target_col], errors="coerce").dropna()

    if values.empty:
        print(f"Skipping 3xIQR target plot for {target_col}: no numeric values")
        return {}

    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - (iqr_multiplier * iqr)
    upper = q3 + (iqr_multiplier * iqr)

    outlier_mask = ~values.between(lower, upper)
    outliers = values.loc[outlier_mask].rename(target_col).to_frame()

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    if not outliers.empty:
        outliers.to_csv(save_path / f"{save_fname}_outliers.csv", index_label="ID")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axvspan(
        lower,
        upper,
        color="tab:green",
        alpha=0.12,
        label=f"{iqr_multiplier:g}x IQR range",
    )
    ax.axvline(lower, color="tab:red", linestyle="--", linewidth=1.2)
    ax.axvline(upper, color="tab:red", linestyle="--", linewidth=1.2)
    ax.axvline(
        values.median(), color="black", linestyle="-", linewidth=1.0, label="Median"
    )

    sns.histplot(values, bins=bins, kde=True, ax=ax)

    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    ax.set_title(
        f"{target_col} Target Distribution with {iqr_multiplier:g}x IQR Bounds"
    )
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    fig.savefig(save_path / f"{save_fname}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "target_n": int(len(values)),
        "target_q1": float(q1),
        "target_q3": float(q3),
        "target_iqr": float(iqr),
        "target_lower_3xIQR": float(lower),
        "target_upper_3xIQR": float(upper),
        "target_n_outliers_3xIQR": int(outlier_mask.sum()),
        "target_outlier_fraction_3xIQR": float(outlier_mask.mean()),
    }


def getLipinskiFilteredExternalPerformanceDf(
    properties: list[str],
    feature_sets: list[str],
    full_pathing: dict,
    preds_dir: dict,
    target_columns: dict[str, str],
) -> pd.DataFrame:
    rows = []

    for prop in properties:
        if prop not in preds_dir:
            print(f"{prop} not in prediction output paths")
            continue

        try:
            rdkit_df = loadFeaturePattern(full_pathing["full_features"][prop]["rdkit"])
            lipinski_ids = pd.Index(checkLipinskiCriteria(rdkit_df)).astype(str)

            target_col = target_columns[prop]
            target_df = pd.read_csv(
                full_pathing["targets"][prop],
                index_col="ID",
            )
            if target_col not in target_df.columns:
                print(f"{target_col} not in target columns for {prop}")
                continue

            target_df = target_df[[target_col]].rename(columns={target_col: "true"})
            target_df.index = target_df.index.astype(str)

        except Exception as e:
            print(f"Could not prepare Lipinski IDs for {prop}: {e}")
            continue

        for feature_set in feature_sets:
            if feature_set not in preds_dir[prop]:
                continue

            pred_path = Path(preds_dir[prop][feature_set]) / "last_20pct_pred.csv.gz"
            if not pred_path.exists():
                print(
                    f"Missing last_20pct predictions for {prop} / {feature_set}: {pred_path}"
                )
                continue

            try:
                pred_df = pd.read_csv(pred_path, index_col=0)
                pred_df.index = pred_df.index.astype(str)

                pred_col = (
                    target_col if target_col in pred_df.columns else pred_df.columns[0]
                )
                pred_df = pred_df[[pred_col]].rename(columns={pred_col: "pred"})

                keep_ids = pred_df.index.intersection(target_df.index).intersection(
                    lipinski_ids
                )

                eval_df = pred_df.loc[keep_ids].join(
                    target_df.loc[keep_ids],
                    how="inner",
                )
                eval_df = eval_df.dropna(subset=["pred", "true"])

                if eval_df.empty:
                    print(f"No Lipinski-matched rows for {prop} / {feature_set}")
                    continue

                rows.append(
                    {
                        "property": prop,
                        "split": "external_lipinski",
                        "feature_set": feature_set,
                        "r2": calculateR2(eval_df["true"], eval_df["pred"]),
                        "pearson_r": eval_df["true"].corr(eval_df["pred"]),
                        "n": len(eval_df),
                        "n_lipinski_ids": len(lipinski_ids),
                    }
                )

            except Exception as e:
                print(
                    f"Could not calculate Lipinski performance for {prop} / {feature_set}: {e}"
                )

    return pd.DataFrame(rows)


def get3xIQRFilteredExternalPerformanceDf(
    properties,
    feature_sets,
    full_pathing,
    preds_dir,
    target_columns,
):
    rows = []

    for prop in properties:
        target_col = target_columns[prop]

        target_df = pd.read_csv(full_pathing["targets"][prop], index_col="ID")
        target_df.index = target_df.index.astype(str)
        true = pd.to_numeric(target_df[target_col], errors="coerce")

        q1 = true.quantile(0.25)
        q3 = true.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - (3 * iqr)
        upper = q3 + (3 * iqr)

        keep_target = true.loc[true.between(lower, upper)].rename("true")

        for feature_set in feature_sets:
            if feature_set not in preds_dir[prop]:
                print(f"{feature_set} not in prediction output paths for {prop}")
                continue

            pred_path = Path(preds_dir[prop][feature_set]) / "last_20pct_pred.csv.gz"

            if not pred_path.exists():
                print(f"Missing predictions for {prop} / {feature_set}: {pred_path}")
                continue

            pred_df = pd.read_csv(pred_path, index_col=0)
            pred_df.index = pred_df.index.astype(str)

            pred_col = (
                target_col if target_col in pred_df.columns else pred_df.columns[0]
            )
            pred = pd.to_numeric(pred_df[pred_col], errors="coerce").rename("pred")

            eval_df = pd.concat([keep_target, pred], axis=1, join="inner").dropna()

            if len(eval_df) < 2:
                continue

            err = eval_df["pred"] - eval_df["true"]

            rows.append(
                {
                    "property": prop,
                    "split": "external_3xIQR",
                    "feature_set": feature_set,
                    "stat": "mean",
                    "r2": calculateR2(eval_df["true"], eval_df["pred"]),
                    "pearson_r": eval_df["true"].corr(
                        eval_df["pred"], method="pearson"
                    ),
                    "rmse": (err.pow(2).mean()) ** 0.5,
                    "bias": err.mean(),
                    "sdep": (err - err.mean()).pow(2).mean() ** 0.5,
                    "n": len(eval_df),
                    "lower_3xIQR": lower,
                    "upper_3xIQR": upper,
                }
            )

    return pd.DataFrame(rows)
