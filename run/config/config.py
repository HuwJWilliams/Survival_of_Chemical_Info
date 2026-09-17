"""
File to hold all of the configuration of the run
"""

# %% ===== Python Imports =====
from pathlib import Path
import sys

# %% ===== Project Imports & Pathing Setup=====
SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"

# %% ===== Configuration variables =====

# --- Pathing JSON variables
PATHING_PATH = Path(SCRIPTS_DIR / "src" / "pathing")
PATHING_JSON_NAME = "paths.json"
PATHING_JSON_PATH = PATHING_PATH / PATHING_JSON_NAME

sys.path.insert(0, str(PATHING_PATH))
from get_paths import getPaths

# Empty pathing initialised
EMPTY_PATHING = {
    "imp_dirs": {},
    "train_test_splits": {},
    "raw_data": {},
    "targets": {},
    "full_features": {},
    "prediction_output_dirs": {
        "rf": {},
        "cross_feature_predictions": {},
        "lipinski_cross_feature_predictions": {},
    },
    "dataset_analysis": {},
    "config": {},
}


def loadPathing(required: bool = False) -> dict:
    if PATHING_JSON_PATH.exists():
        return getPaths(PATHING_JSON_PATH)

    if required:
        raise FileNotFoundError(
            f"Pathing JSON has not been created yet: {PATHING_JSON_PATH}"
        )

    return EMPTY_PATHING.copy()


FULL_PATHING = loadPathing()

# --- Target Columns
TARGET_COLUMNS = {
    "bp": "Boiling_Point",
    "logd": "LogD",
    "pka": "pKa",
    "pka_paper1_basic": "pKa",
    "pka_paper1_acidic": "pKa",
    "log_ld50": "LOG_LD50",
    "pic50": "pIC50",
    "hole_re": "Hole_Reorganisation_Energy",
    "elec_re": "Electron_Reorganisation_Energy",
    "aq_sol": "Solubility",
    "egfr_pic50": "pIC50",
}


# --- Supported Feature Sets
SUPPORTED_FEATURE_SETS = (
    "rdkit",
    "mordred",
    "morgan",
    "maccs",
    "chemberta-dc",
    "chemberta-sey",
    "molformer-ibm",
    "molformer-dc",
    "selformer",
    "ft-scaffold-chemberta-dc",
    "ft-scaffold-chemberta-sey",
    "ft-scaffold-molformer-ibm",
    "ft-scaffold-molformer-dc",
    "ft-scaffold-selformer",
    "ft-random-chemberta-dc",
    "ft-random-chemberta-sey",
    "ft-random-molformer-ibm",
    "ft-random-molformer-dc",
    "ft-random-selformer",
)

# --- Supported Target Sets
SUPPORTED_TARGET_SETS = tuple(FULL_PATHING.get("targets", {}).keys()) + ("all",)

# --- Transformer Specifications
TRANSFORMER_FEATURE_SPECS: dict[str, dict[str, str]] = {
    "chemberta-dc": {
        "tokeniser": "DeepChem/ChemBERTa-100M-MLM",
        "model": "DeepChem/ChemBERTa-100M-MLM",
        "model_label": "ChemBERTa-DC",
        "suffix_label": "chemberta-dc",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "f5c45f44d3061f0346888f5c09db17ec1146d29d",
        "max_token_len": 512,
    },
    "chemberta-sey": {
        "tokeniser": "seyonec/ChemBERTa-zinc-base-v1",
        "model": "seyonec/ChemBERTa-zinc-base-v1",
        "model_label": "ChemBERTaSey",
        "suffix_label": "chemberta-sey",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "761d6a18cf99db371e0b43baf3e2d21b3e865a20",
        "max_token_len": 512,
    },
    "molformer-ibm": {
        "tokeniser": "ibm/MoLFormer-XL-both-10pct",
        "model": "ibm/MoLFormer-XL-both-10pct",
        "model_label": "MolFormer-IBM",
        "suffix_label": "molformer-ibm",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "7b12d946c181a37f6012b9dc3b002275de070314",
        "max_token_len": 202,
    },
    "molformer-dc": {
        "tokeniser": "DeepChem/MoLFormer-c3-1.1B",
        "model": "DeepChem/MoLFormer-c3-1.1B",
        "model_label": "MolFormer-DC",
        "suffix_label": "molformer-dc",
        "input_kind": "smiles",
        "metadata_col_name": "SMILES",
        "commit_hash": "9f1b9ea3590833bd0ea1a70e789c5d3da11ba7ed",
        "code_revision": "refs/pr/1",
        "max_token_len": 202,
    },
    "selformer": {
        "tokeniser": "HUBioDataLab/SELFormer",
        "model": "HUBioDataLab/SELFormer",
        "model_label": "SELFormer",
        "suffix_label": "selformer",
        "input_kind": "selfies",
        "metadata_col_name": "SELFIES",
        "commit_hash": "177d98b158e999a6cb7fc9743dbfe1e8a17c57e5",
        "max_token_len": 512,
    },
}


CFP_ANALYSIS_METRICS = {
    "regression": {
        "metric": "Pearson_r",
        "bar_metrics": ["Pearson_r"],
        "group_metrics": ["Pearson_r", "r2", "RMSE", "Bias"],
        "member_suffix": "reg",
        "radar_metrics": ["avg_Pearson_r"],
    },
    "binary_classification": {
        "metric": "Balanced_Accuracy",
        "bar_metrics": ["AUC", "MCC", "Balanced_Accuracy"],
        "group_metrics": [
            "Accuracy",
            "Sensitivity",
            "Specificity",
            "PPV",
            "NPV",
            "AUC",
            "MCC",
            "Balanced_Accuracy",
        ],
        "member_suffix": "cla",
        "radar_metrics": ["avg_AUC"],
    },
    "multiclass_classification": {
        "metric": "Balanced_Accuracy",
        "bar_metrics": ["AUC_OVR", "MCC", "Balanced_Accuracy"],
        "group_metrics": [
            "Accuracy",
            "Balanced_Accuracy",
            "F1_macro",
            "AUC_OVR",
            "MCC",
        ],
        "member_suffix": "mcla",
        "radar_metrics": ["avg_AUC_OVR"],
    },
}

PP_ANALYSIS_METRICS = {
    "regression": {
        "metric": "Pearson_r",
        "bar_metrics": ["Pearson_r"],
    },
}
