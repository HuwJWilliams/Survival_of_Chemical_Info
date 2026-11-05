"""
Script to train Random Forest Regressors on one set of features to predict another
(e.g., train on ChemBERTa embeddings to predict RDKit descriptors)
"""

# %% ===== Python Imports =====
import argparse
import pandas as pd
from pathlib import Path
import sys
from glob import glob
import json

# %% ===== Project Imports & Pathing Setup =====
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(RUN_DIR / "config"))

from config import (
    SRC_DIR,
    SUPPORTED_FEATURE_SETS,
    PATHING_JSON_PATH,
    SUPPORTED_TARGET_SETS,
)

sys.path.insert(0, str(SRC_DIR / "models"))
from transfer_model import TL

sys.path.insert(0, str(SRC_DIR / "pathing"))
from get_paths import getPaths

FULL_PATHING = getPaths(PATHING_JSON_PATH)

sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_cleaning import cleanFeatureDF

FULL_PATHING = getPaths(PATHING_JSON_PATH)

# %% ===== Argument Parsing =====
parser = argparse.ArgumentParser(description="Generating cross-feature predictions")

parser.add_argument(
    "--train",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to train RF models on. Choices are:\n" f"{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument(
    "--target",
    required=True,
    choices=SUPPORTED_FEATURE_SETS,
    help="Features to test RF models on. Choices are:\n" f"{SUPPORTED_FEATURE_SETS}",
)

parser.add_argument("--property", default="all", choices=SUPPORTED_TARGET_SETS)

parser.add_argument(
    "--save-dir",
    default="cross_feature_predictions",
    help="Directory to save the results to",
)

parser.add_argument(
    "--model-name",
    type=str,
    default="random_forest",
    help="Model to train using the configured hyperparameters",
)

parser.add_argument(
    "--n-resamples", type=int, default=1, help="Number of resamples in the outer loop"
)

parser.add_argument(
    "--test-size",
    type=float,
    default=0.3,
    help="Fraction of the training set (0.3 = 30 %)",
)

parser.add_argument(
    "--skip-existing",
    action="store_true",
    help="Flag to skip already processed features",
)

parser.add_argument(
    "--save-models", action="store_true", help="Flag to save trained models"
)

parser.add_argument(
    "--shuffle-data",
    action="store_true",
    help="Flag to shuffle the data prior to training",
)

parser.add_argument(
    "--save-feat-imp", action="store_true", help="Flag to save feature importance data"
)

parser.add_argument(
    "--lipinski-mols",
    action="store_true",
    help="Flag to do cross-feature predictions on Lipinksi-fitting molecules only \
        (if created using join_all_target_molecule_datasets.py, otherwise it will crash)",
)

parser.add_argument(
    "--minimum-targs",
    type=int,
    default=2500,
    help="Minimum number of samples to train RF models",
)

args = parser.parse_args()
train = args.train.lower()
target = args.target.lower()
identifier = f"pred_{target}_tr_{train}"

if args.lipinski_mols and args.property != "all":
    raise ValueError("--lipinski-mols cannot be used with a property-specific dataset.")
property_dataset = args.property.lower()
mols = "fit_lipinski" if args.lipinski_mols else property_dataset

full_feats = FULL_PATHING["full_features"][mols]

save_dir = args.save_dir
save_path = FULL_PATHING["prediction_output_dirs"][save_dir][property_dataset][
    identifier
]

model_name = args.model_name

# Loading relevant hyperparameters and saving to result path
hyperparameter_path = RUN_DIR / "model_hyperparameters.json"
with open(hyperparameter_path, "r") as f:
    hyperparameter_config = json.load(f)
if model_name not in hyperparameter_config:
    raise ValueError(f"No hyperparameters configured for model '{model_name}'")
hyper_params = hyperparameter_config[model_name]

save_path.mkdir(parents=True, exist_ok=True)
with open(save_path / f"{model_name}_hyperparameters.json", "w") as f:
    json.dump(hyper_params, f, indent=4)


# %% ===== Helper Functions =====
def getFeatureFiles(feature_path: str | Path) -> list[Path]:
    feature_path = Path(feature_path)

    if "*" not in str(feature_path):
        if not feature_path.exists():
            raise FileNotFoundError(f"Feature file not found: {feature_path}")
        return [feature_path]

    feature_files = [Path(file) for file in sorted(glob(str(feature_path)))]

    # Some older generated files are unbatched, e.g. name.csv instead of name_1.csv.
    if not feature_files and str(feature_path).endswith("_*.csv"):
        unbatched_path = Path(str(feature_path).replace("_*.csv", ".csv"))
        if unbatched_path.exists():
            feature_files = [unbatched_path]

    if not feature_files:
        raise FileNotFoundError(f"No feature files matched: {feature_path}")

    return feature_files


def loadAndCleanData(
    name: str,
    full_feats: dict,
    correlation_threshold: float | None = None,
) -> pd.DataFrame:
    if name not in full_feats:
        raise KeyError(f"{name} not found in full_features[{mols}]")

    feature_files = getFeatureFiles(full_feats[name])
    df = pd.concat(
        [pd.read_csv(file, index_col="ID", low_memory=False) for file in feature_files],
        axis=0,
    )

    df, clean_report = cleanFeatureDF(
        df,
        max_nan_fraction=0.10,
        drop_constant_cols=True,
        median_impute=False,
        correlation_threshold=correlation_threshold,
    )

    print(f"\nFeature cleaning report for {name}:")
    print(clean_report)
    print(f"Final shape: {df.shape}")

    return df


# %% ===== Running Cross-Feature Prediction Experiments =====
train_df = loadAndCleanData(train, full_feats=full_feats, correlation_threshold=None)

target_df = loadAndCleanData(target, full_feats=full_feats, correlation_threshold=None)

common_idx = train_df.index.intersection(target_df.index)
train_df, target_df = train_df.loc[common_idx], target_df.loc[common_idx]

print(f"[Multi-Target RF] mols={mols}, train={train}, test={target}, id={identifier}")
print(f"Train shape: {train_df.shape}, Test shape: {target_df.shape}")
print(f"Save path: {save_path}")

model = TL(
    log_to_file=False,
    log_identifier=identifier,
)

model.trainMultiTargetModels(
    features_df=train_df,
    targets_df=target_df,
    model_name=model_name,
    hyper_params=hyper_params,
    output_csv=f"{identifier}.csv",
    existing_performance_csv=save_path / f"{identifier}.csv",
    n_resamples=args.n_resamples,
    test_size=args.test_size,
    save_path=save_path,
    skip_existing=args.skip_existing,
    save_models=args.save_models,
    save_feat_imp=args.save_feat_imp,
    min_training_samples=args.minimum_targs,
)
