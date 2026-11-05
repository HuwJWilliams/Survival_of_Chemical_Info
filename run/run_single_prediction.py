"""
Script to train, run, and save predictions using various regressor models
"""

# %% ===== Python Imports =====
import argparse
import json
import pandas as pd
from pathlib import Path
import sys
from glob import glob

# %% ===== Project Imports & Pathing Setup =====
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR / "config"))

from config import (
    SCRIPTS_DIR,
    SRC_DIR,
    PATHING_JSON_PATH,
    SUPPORTED_FEATURE_SETS,
    TARGET_COLUMNS,
)

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData

sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_cleaning import cleanFeatureDF
from group_descriptors import getGroups

# %% ===== Argument Parsing =====
p = argparse.ArgumentParser()

p.add_argument(
    "--model-name",
    type == "str",
    required=True,
    help="Model type to use. E.g., random_forest, svm, ridge, mlp",
)

p.add_argument(
    "--predict-on", required=True, choices=list(FULL_PATHING["targets"].keys())
)
p.add_argument("--feature-set", required=True, choices=SUPPORTED_FEATURE_SETS)
p.add_argument("--target-column", default=None, help="Override the target column name")
p.add_argument("--identifier", default=None, help="Run identifier suffix (optional)")
p.add_argument(
    "--repeats", type=int, default=1, help="Number of independent repeats to run"
)
p.add_argument(
    "--base-seed",
    type=int,
    default=42,
    help="Base random seed used to seed each repeat",
)
p.add_argument(
    "--n-resamples",
    type=int,
    default=50,
    help="Number of internal resamples per repeat",
)
p.add_argument(
    "--max-nan-frac",
    type=float,
    default=0,
    help="Fraction of NaN rows to drop a column",
)
p.add_argument(
    "--corr-threshold",
    type=float,
    default=0.9,
    help="Correlation threshold for training features",
)

p.add_argument(
    "--additional-features",
    type=json.loads,
    default=None,
    help=(
        "Optional JSON dict of additional features. Example: "
        '\'{"feature_set": "mordred", "individual_features": ["ABC"], '
        '"feature_group": ["Autocorrelation"]}\''
    ),
)


def safe_path_label(value: str) -> str:
    return "".join(
        char if char.isalnum() or char in ("-", "_") else "-" for char in str(value)
    ).strip("-")


def getAdditionalFeaturesOutDirName(add_feats: dict) -> str:
    feature_set = safe_path_label(add_feats["feature_set"])
    individual_feats = add_feats.get("individual_features", [])
    feat_groups = add_feats.get("feature_group", [])

    if isinstance(individual_feats, str):
        individual_feats = [individual_feats]

    if isinstance(feat_groups, str):
        feat_groups = [feat_groups]

    out_dir_parts = [feature_set]

    if feat_groups:
        out_dir_parts.append(
            "groups-" + "-".join(safe_path_label(group) for group in feat_groups)
        )

    if individual_feats:
        out_dir_parts.append(
            "features-"
            + "-".join(safe_path_label(feature) for feature in individual_feats)
        )

    return "__".join(out_dir_parts)


args = p.parse_args()

model_name = args.model_name
task = args.predict_on
feat = args.feature_set
target_col = args.target_column or TARGET_COLUMNS[task]
identifier = args.identifier or f"{task}_{feat}"
max_nan_frac = args.max_nan_frac
corr_threshold = args.corr_threshold
in_features = FULL_PATHING["full_features"][task][feat]
in_targets = FULL_PATHING["targets"][task]
out_dir = FULL_PATHING["prediction_output_dirs"][model_name][task][feat]
add_feats = args.additional_features

if add_feats is not None and add_feats.get("feature_set") is not None:
    additional_features_out_dir = getAdditionalFeaturesOutDirName(add_feats)
    out_dir = out_dir / "additional_features" / additional_features_out_dir
    identifier = f"{identifier}_plus_{additional_features_out_dir}"

out_dir.mkdir(parents=True, exist_ok=True)

if add_feats is not None and add_feats.get("feature_set") is not None:
    with open(out_dir / "additional_features_config.json", "w") as f:
        json.dump(add_feats, f, indent=4)

        hyperparameter_path = RUN_DIR / "model_hyperparameters.json"

with open(hyperparameter_path, "r") as f:
    hyperparameter_config = json.load(f)

if model_name not in hyperparameter_config:
    raise ValueError(f"No hyperparameters configured for model '{model_name}'")

hyper_params = hyperparameter_config[model_name]

print(f"[{model_name.upper()}] task={task}  feature={feat}")
print(f"Features: {in_features}")
print(f"Targets : {in_targets}")
print(f"Output  : {out_dir}")
print(f"Target column: {target_col}")

# %%F ===== Setting Up Data =====

X = loadData(in_features, index_col="ID", wildcard="*")

if add_feats is not None and add_feats.get("feature_set") is not None:
    add_feature_path = FULL_PATHING["full_features"][task][add_feats["feature_set"]]

    add_feature_files = sorted(glob(str(add_feature_path)))

    if not add_feature_files:
        raise FileNotFoundError(
            f"No files found for additional features: {add_feature_path}"
        )

    additional_feats_df = pd.concat(
        [pd.read_csv(file, index_col=0) for file in add_feature_files],
        axis=0,
    )

    individual_feats = add_feats.get("individual_features", [])
    feat_groups = add_feats.get("feature_group", [])

    selected_additional_feats = []

    if individual_feats:
        selected_additional_feats.extend(individual_feats)

    if isinstance(feat_groups, str):
        feat_groups = [feat_groups]

    for feat_group in feat_groups:
        selected_additional_feats.extend(
            getGroups(add_feats["feature_set"])[feat_group]
        )

    # If only feature_set is defined, use all columns from that feature set
    if not selected_additional_feats:
        selected_additional_feats = additional_feats_df.columns.tolist()

    selected_additional_feats = list(dict.fromkeys(selected_additional_feats))
    missing_additional_feats = [
        feature
        for feature in selected_additional_feats
        if feature not in additional_feats_df.columns
    ]

    if missing_additional_feats:
        raise SystemExit(
            "Additional feature columns not found in "
            f"{add_feature_path}: {missing_additional_feats}"
        )

    additional_feats_df = additional_feats_df[selected_additional_feats]
    additional_feats_df = additional_feats_df.add_prefix(
        f"add_{add_feats['feature_set']}_"
    )

    X = X.join(additional_feats_df, how="left")
    print(f"Added additional features: {len(selected_additional_feats)}")

X, clean_report = cleanFeatureDF(
    X,
    max_nan_fraction=max_nan_frac,
    drop_constant_cols=True,
    median_impute=False,
    correlation_threshold=corr_threshold,
)

print("Feature cleaning report:")
print(f"  Dropped metadata cols: {len(clean_report['dropped_metadata'])}")
print(
    f"  Dropped >{max_nan_frac * 100}% NaN cols: {len(clean_report.get('dropped_high_nan_cols', []))}"
)
print(f"  Dropped constant cols: {len(clean_report['dropped_constant_cols'])}")
print(f"  Median-imputed cols: {len(clean_report['median_imputed_cols'])}")
print(f"  Dropped correlated cols: {len(clean_report['dropped_correlated_cols'])}")

y_df = pd.read_csv(in_targets, index_col="ID")
if target_col not in y_df.columns:
    raise SystemExit(
        f"Target column '{target_col}' not found in {in_targets}. "
        f"Available: {list(y_df.columns)}"
    )

y = y_df[[target_col]]

# Align Indexes
common = X.index.intersection(y.index)
X = X.loc[common]
y = y.loc[common]

print(f"Aligned shapes → X: {X.shape}, y: {y.shape}")

data = X.join(y)
print(f"Full modelling data: {data.shape}")

# %% ===== Setting up training resamples =====

repeat_perf_records = []
repeat_internal_records = []
repeat_pred_series = []

for repeat_i in range(args.repeats):
    repeat_n = repeat_i + 1
    repeat_seed = args.base_seed + repeat_i
    repeat_dir = out_dir / "repeats" / f"repeat_{repeat_n:03d}"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRepeat {repeat_n}/{args.repeats} | seed={repeat_seed}")

    # Shuffling Data & making TR/TE split
    shuffled_data = data.sample(frac=1.0, random_state=repeat_seed)
    cut = int(0.8 * len(shuffled_data))
    train_data = shuffled_data.iloc[:cut].copy()
    test_data = shuffled_data.iloc[cut:].copy()
    print(f"Train: {train_data.shape}  Test: {test_data.shape}")

    # Saving TR/TE Data
    split_dir = repeat_dir / "training_data"
    split_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"ID": train_data.index}).to_csv(
        split_dir / "split_train_ids.csv", index=False
    )
    pd.DataFrame({"ID": test_data.index}).to_csv(
        split_dir / "split_validation_ids.csv", index=False
    )

    # %% ===== Training Models =====
    model = TL(log_identifier=f"{identifier}_repeat_{repeat_n:03d}")
    final_model, _, internal_perf, _, _ = model.trainSingleTargetModel(
        data=train_data,
        target_column=target_col,
        model_name=model_name,
        hyper_params=hyper_params,
        n_resamples=args.n_resamples,
        test_size=0.3,
        random_seed=repeat_seed,
        save_models=True,
        save_path=repeat_dir,
    )

    _, _, external_perf = model.predictSingleTarget(
        model=final_model,
        model_name=model_name,
        task_type="regression",
        data=test_data,
        target_column=target_col,
        calc_perf=True,
        save_preds=True,
        save_path=repeat_dir,
        preds_filename="external_preds",
        perf_filename="external_perf",
    )

    # %% ===== Recording and Averaging resample performances =====
    repeat_perf = {"repeat": repeat_n, "seed": repeat_seed, **external_perf}
    repeat_perf_records.append(repeat_perf)

    repeat_internal_records.append(
        {
            "repeat": repeat_n,
            "seed": repeat_seed,
            **internal_perf,
        }
    )

    (repeat_dir / "external_perf.json").unlink(missing_ok=True)
    (repeat_dir / f"{target_col}_internal_performance_dict.json").unlink(
        missing_ok=True
    )
    (repeat_dir / "training_data" / "performance_stats.json").unlink(missing_ok=True)

    pred_name = f"repeat_{repeat_n:03d}"
    saved_pred_df = pd.read_csv(repeat_dir / "external_preds.csv.gz", index_col="ID")
    repeat_pred_series.append(saved_pred_df[target_col].rename(pred_name))


def summarise_performance(records: list[dict]) -> dict:
    perf_df = pd.DataFrame(records).set_index("repeat")
    numeric_perf = perf_df.select_dtypes(include="number").drop(
        columns=["seed"], errors="ignore"
    )

    return {
        "individual": perf_df.reset_index().to_dict(orient="records"),
        "mean": {
            metric: round(float(value), 6)
            for metric, value in numeric_perf.mean(axis=0).items()
        },
        "std": {
            metric: round(float(value), 6)
            for metric, value in numeric_perf.std(axis=0).items()
        },
    }


model_performance = {
    "model_name": model_name,
    "n_repeats": args.repeats,
    "base_seed": args.base_seed,
    "internal": summarise_performance(repeat_internal_records),
    "external": summarise_performance(repeat_perf_records),
}

with open(out_dir / f"{model_name}_performance.json", "w") as f:
    json.dump(model_performance, f, indent=4)

if repeat_pred_series:
    pred_df = pd.concat(repeat_pred_series, axis=1)
    pred_df[target_col] = pred_df.mean(axis=1)
    pred_df["n_repeats_predicted"] = (
        pred_df[[f"repeat_{repeat_i + 1:03d}" for repeat_i in range(args.repeats)]]
        .notna()
        .sum(axis=1)
    )
    pred_df.to_csv(
        out_dir / "last_20pct_pred_all_repeats.csv.gz",
        index_label="ID",
        compression="gzip",
    )
    pred_df[[target_col, "n_repeats_predicted"]].to_csv(
        out_dir / "last_20pct_pred.csv.gz",
        index_label="ID",
        compression="gzip",
    )
