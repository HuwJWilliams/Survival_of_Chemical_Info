"""
Separated script to run any of the feature importance analysis defined in src/visualisation/vis.py
"""

# region Imports and Pathing
import sys
import pandas as pd
from pathlib import Path
import argparse
import joblib
import matplotlib.pyplot as plt

# --- Paths
FILE_DIR = Path(__file__).parent
SCRIPTS_DIR = FILE_DIR.parent
SRC_DIR = SCRIPTS_DIR / "src"

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

PATHS = getPaths()

sys.path.insert(0, str(SRC_DIR / "visualisation"))
from vis import Visualise

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import getMostImportantFeatures

sys.path.insert(0, str(SRC_DIR / "datasets"))
from group_descriptors import getGroups

# endregion

# region Class and Parser Setup
v = Visualise()

parser = argparse.ArgumentParser(description="Generating Feature Importance analysis")
# endregion

parser.add_argument(
    "--targ",
    required=True,
    help="Target prediction set",
)

parser.add_argument(
    "--imp",
    required=True,
    help="Prediction set to check importance for",
)

parser.add_argument(
    "--test",
    required=True,
    help="Test set to check both target and predictions on important features",
)

parser.add_argument(
    "--imp-type",
    required=True,
    help="Type of Feature Importance being used. Options: rf, shap, both",
)

parser.add_argument(
    "--results-dir",
    default="lipinski_cross_feature_predictions",
    help="Directory holding all of the prediction directories",
)

parser.add_argument(
    "--descriptor-group", default="rdkit", help="Feature set to put into groups"
)

parser.add_argument(
    "--save-path",
    default=PATHS["imp_dirs"]["results_dir"] / "top_25_importance_feature_predictions",
)

parser.add_argument(
    "--mode",
    default="avg",
    help="Set 'avg' for feature importance averaging, or 'cum' for cumulative feature importance",
)

parser.add_argument(
    "--min-max-high",
    nargs=3,
    type=int,
    default=(0, 1, 5),
    help="Set the minimumm and maximum and outlying boundary when colouring by importance",
)

parser.add_argument(
    "--bubble",
    action="store_true",
    help="Flag to set left plot to increase size with feature importance",
)

parser.add_argument(
    "--x-metric", default="AUC", help="Performance metric on the X-axis of plots"
)

parser.add_argument(
    "--y-metric", default="Pearson_r", help="Performance metric on the Y-axis of plots"
)

args = parser.parse_args()

targ = args.targ
imp = args.imp
test = args.test
imp_type = args.imp_type.upper()
mode = args.mode
save_path = Path(args.save_path)
save_path.mkdir(parents=True, exist_ok=True)

targ_exp = f"pred_{targ}_tr_{test}"
imp_exp = f"pred_{targ}_tr_{imp}"
test_exp = f"pred_{imp}_tr_{test}"

PRED_PATHS = PATHS["prediction_output_dirs"][args.results_dir]

pred_targ_tr_test = pd.read_csv(PRED_PATHS[targ_exp] / f"{targ_exp}.csv", index_col=0)
pred_targ_tr_imp = pd.read_csv(PRED_PATHS[imp_exp] / f"{imp_exp}.csv", index_col=0)
pred_imp_tr_test = pd.read_csv(PRED_PATHS[test_exp] / f"{test_exp}.csv", index_col=0)

shap_bundle = joblib.load(PRED_PATHS[imp_exp] / "shap" / "full_shap_analysis.joblib.gz")
rf_fi_df = pd.read_csv(PRED_PATHS[imp_exp] / "all_feature_importance.csv", index_col=0)

group_map = getGroups(args.descriptor_group)
desc_to_group = {
    desc: group for group, members in group_map.items() for desc in members
}
unknown = [d for d in pred_targ_tr_imp.index if d not in desc_to_group]
print(f"Unknown descriptors: {len(unknown)}")
print(unknown[:100])

shap_importance_source = {
    "shap_by_desc": shap_bundle["shap_by_desc"],
    "feature_names": shap_bundle["feat_explain"].columns.tolist(),
}

shap_avg_imp, shap_cum_imp, imp_feat_shap_count = getMostImportantFeatures(
    shap_importance_source, mode="shap"
)
rf_avg_imp, rf_cum_imp, imp_feat_rf_count = getMostImportantFeatures(
    rf_fi_df, mode="rf"
)

importance_lookup = {
    ("avg", "SHAP"): shap_avg_imp,
    ("avg", "RF"): rf_avg_imp,
    ("cum", "SHAP"): shap_cum_imp,
    ("cum", "RF"): rf_cum_imp,
}
importance_map = importance_lookup[(mode, imp_type)]


shap_avg_df = v.plotDescPredictionVsFeatPrediction(
    importance_map=importance_map,
    pred_tr_perf_df=pred_imp_tr_test,
    pred_on_target_df=pred_targ_tr_imp,
    desc_to_group=desc_to_group,
    importance_col=f"avg_imp_top25_{imp}",
    left_title=f"{args.imp_type} {mode} importance: Colored by {args.descriptor_group} group",
    right_title=f"{imp_type} {mode} importance: Colored by importance",
    save_path=save_path / f"{test}_vs_{imp_type}_importance_{imp}_{mode}.png",
    mode=mode,
    r_vmin=args.min_max_high[0],
    r_vmax=args.min_max_high[1],
    r_high=args.min_max_high[2],
    imp_type=args.imp_type,
    bubble=args.bubble,
    x_metric_col=args.x_metric,
    y_metric_col=args.y_metric,
)

# Plotting Imp preds vs Test preds
plot_df = (
    pred_targ_tr_imp[[args.x_metric]]
    .rename(columns={args.x_metric: f"{imp}_on_{targ}"})
    .join(
        pred_targ_tr_test[[args.y_metric]].rename(
            columns={args.y_metric: f"{test}_on_{targ}"}
        ),
        how="inner",
    )
    .dropna()
)

plt.figure(figsize=(7, 7))
plt.scatter(
    plot_df[f"{imp}_on_{targ}"],  # x
    plot_df[f"{test}_on_{targ}"],  # y
    s=25,
    alpha=0.7,
    edgecolor="none",
)

plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.xlabel(f"{imp} -> {targ} {args.x_metric}")
plt.ylabel(f"{test} -> {targ} {args.y_metric}")
plt.title(f"Descriptor-level performance: {imp} vs {test}")
plt.tight_layout()
plt.savefig(save_path / f"{test}_vs_{imp}_pred_{targ}_scatter.png", dpi=300)
plt.close()


def _plotTopFeatureCount(
    count_map: dict, method_label: str, out_path: Path, top_n: int = 25
) -> list[str]:
    if not count_map:
        print(f"No features found for {method_label} histogram; skipping.")
        return []

    feature_counts = {
        str(feature): int(count)
        for feature, count in count_map.items()
        if int(count) > 0
    }
    if not feature_counts:
        print(f"No non-zero feature counts for {method_label}; skipping histogram.")
        return []

    sorted_items = sorted(feature_counts.items(), key=lambda kv: kv[1], reverse=True)
    features = [k for k, _ in sorted_items]
    counts = [v for _, v in sorted_items]

    total_features = len(counts)
    thresholds = [20, 40, 60, 80, 100]
    print(
        f"\n{method_label} top-{top_n} feature frequency summary (n={total_features} features):"
    )
    for thr in thresholds:
        n_above = sum(c > thr for c in counts)
        pct = (100.0 * n_above / total_features) if total_features else 0.0
        print(f"  > {thr}: {n_above}/{total_features} ({pct:.2f}%)")

    plt.figure(figsize=(max(10, len(features) * 0.22), 6))
    plt.bar(features, counts, edgecolor="black")
    plt.xticks(rotation=90)
    plt.xlabel("Feature name")
    plt.ylabel(f"Count in top-{top_n}")
    plt.title(f"{method_label}: Feature frequency in top-{top_n} importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    return features


def _plotTopFeaturePredictions(
    pred_df: pd.DataFrame,
    features: list[str],
    method_label: str,
    out_path: Path,
    top_n: int = 25,
) -> None:
    if not features:
        print(f"No features passed for {method_label} prediction bar plot; skipping.")
        return

    trimmed_pred_df = pred_df.reindex(features).dropna(subset=["Pearson_r"])
    if trimmed_pred_df.empty:
        print(
            f"No matching Pearson_r rows for {method_label}; skipping prediction bar plot."
        )
        return
    plot_features = trimmed_pred_df.index.astype(str).tolist()

    plt.figure(figsize=(max(10, len(plot_features) * 0.22), 6))
    plt.bar(plot_features, trimmed_pred_df["Pearson_r"], edgecolor="black")
    plt.xticks(rotation=90)
    plt.xlabel("Feature name")
    plt.ylabel(f"Pearson R for Important Features in top-{top_n}")
    plt.title(f"{method_label}: Prediction for Important Features in top-{top_n}")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    return


shap_features = _plotTopFeatureCount(
    count_map=imp_feat_shap_count,
    method_label="SHAP",
    out_path=save_path / f"{imp_exp}_shap_top25_feature_frequency_bar.png",
    top_n=25,
)
rf_features = _plotTopFeatureCount(
    count_map=imp_feat_rf_count,
    method_label="RF",
    out_path=save_path / f"{imp_exp}_rf_top25_feature_frequency_bar.png",
    top_n=25,
)

_plotTopFeaturePredictions(
    pred_df=pred_imp_tr_test,
    features=shap_features,
    method_label="SHAP",
    out_path=save_path / f"{test_exp}_train_shap_top25_feature_prediction_bar.png",
    top_n=25,
)
_plotTopFeaturePredictions(
    pred_df=pred_imp_tr_test,
    features=rf_features,
    method_label="RF",
    out_path=save_path / f"{test_exp}_rf_top25_feature_prediction_bar.png",
    top_n=25,
)
