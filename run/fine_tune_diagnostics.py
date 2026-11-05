"""
Run transformer fine-tuning diagnostics without generating final embeddings.

Example
-------
python run/fine_tune_diagnostics.py ^
    --task bp ^
    --feature-set molformer-dc ^
    --split-by random ^
    --n-mols-fine-tune 5000 ^
    --fine-tune-epochs 40 ^
    --output-dir fine_tuning_diagnostics/bp_molformer_5000mol_40epoch
"""

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUN_DIR = Path(__file__).resolve().parent
PROJ_DIR = RUN_DIR.parent

sys.path.insert(0, str(RUN_DIR / "config"))
from config import (  # noqa: E402
    PATHING_JSON_PATH,
    PATHING_PATH,
    SRC_DIR,
    TARGET_COLUMNS,
    TRANSFORMER_FEATURE_SPECS,
)

sys.path.insert(0, str(SRC_DIR / "datasets"))
from feature_generator import FeatureGenerator  # noqa: E402

sys.path.insert(0, str(PATHING_PATH))
from get_paths import getPaths  # noqa: E402

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import splitByMurckoScaffold  # noqa: E402

FEATURE_SET_ALIASES = {
    "molformer": "molformer-dc",
    "molformer-c3-1b": "molformer-dc",
    "chemberta": "chemberta-dc",
}


def parse_args() -> argparse.Namespace:
    feature_choices = sorted(set(TRANSFORMER_FEATURE_SPECS).union(FEATURE_SET_ALIASES))

    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a transformer model and save only diagnostics "
            "(log history, plots, metrics), not final embeddings."
        )
    )
    parser.add_argument("--task", required=True, help="Dataset key, e.g. bp.")
    parser.add_argument(
        "--feature-set",
        required=True,
        choices=feature_choices,
        help="Base transformer feature set to fine-tune.",
    )
    parser.add_argument(
        "--target-col",
        default=None,
        help="Target column. Defaults to TARGET_COLUMNS[task] from config.py.",
    )
    parser.add_argument(
        "--input-csv",
        default=None,
        help="Optional target CSV path. Defaults to paths.json targets[task].",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for diagnostics. Defaults to fine_tuning_diagnostics/...",
    )
    parser.add_argument(
        "--fine-tune-test-frac",
        type=float,
        default=0.2,
        help="Validation fraction used during fine-tuning.",
    )
    parser.add_argument(
        "--fine-tune-epochs",
        type=float,
        default=40,
        help="Number of fine-tuning epochs.",
    )
    parser.add_argument(
        "--fine-tune-lr",
        type=float,
        default=2e-5,
        help="Fine-tuning learning rate.",
    )
    parser.add_argument(
        "--fine-tune-batch-size",
        type=int,
        default=32,
        help="Per-device train/eval batch size.",
    )
    parser.add_argument(
        "--fine-tune-early-stopping-patience",
        type=int,
        default=5,
        help="Stop after this many evals without improvement. Use 0 to disable.",
    )
    parser.add_argument(
        "--split-by",
        default="random",
        choices=("random", "scaffold"),
        help="How to select molecules for fine-tuning.",
    )
    parser.add_argument(
        "--n-mols-fine-tune",
        type=int,
        default=5000,
        help="Number of molecules to fine-tune on.",
    )
    parser.add_argument(
        "--fine-tune-random-state",
        type=int,
        default=42,
        help="Random seed used when selecting molecules.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "cls"),
        default="mean",
        help="Stored for compatibility with fineTuneTransformer.",
    )
    parser.add_argument("--pathing-json", default=PATHING_JSON_PATH)
    parser.add_argument(
        "--keep-checkpoints",
        action="store_true",
        help="Keep Hugging Face checkpoint-* directories after diagnostics.",
    )

    return parser.parse_args()


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serialisable")


def resolve_output_dir(args: argparse.Namespace, task: str, feature_set: str) -> Path:
    if args.output_dir is not None:
        return Path(args.output_dir)

    epochs = str(args.fine_tune_epochs).replace(".", "p")
    return (
        PROJ_DIR
        / "fine_tuning_diagnostics"
        / (
            f"{task}_{args.split_by}_{feature_set}_"
            f"{args.n_mols_fine_tune}mols_{epochs}epochs"
        )
    )


def load_target_data(args: argparse.Namespace, task: str) -> pd.DataFrame:
    if args.input_csv is not None:
        input_csv = Path(args.input_csv)
    else:
        full_pathing = getPaths(args.pathing_json)
        input_csv = Path(full_pathing["targets"][task])

    data = pd.read_csv(input_csv, index_col="ID")
    if "SMILES" not in data.columns:
        raise KeyError(f"Input file must contain a 'SMILES' column: {input_csv}")

    return data


def select_fine_tuning_rows(
    data: pd.DataFrame,
    target_col: str,
    split_by: str,
    n_mols: int,
    random_state: int,
) -> pd.DataFrame:
    fine_tune_df = data[["SMILES", target_col]].dropna().copy()

    if split_by == "random":
        return fine_tune_df.sample(
            n=min(n_mols, len(fine_tune_df)),
            random_state=random_state,
        )

    random.seed(random_state)
    return splitByMurckoScaffold(
        fine_tune_df,
        n_cmpds=n_mols,
        max_scaff_diversity_pass=5,
    )


def remove_checkpoints(output_dir: Path) -> None:
    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir)


def main() -> None:
    args = parse_args()

    task = args.task.lower()
    feature_set = FEATURE_SET_ALIASES.get(
        args.feature_set.lower(),
        args.feature_set.lower(),
    )
    target_col = args.target_col or TARGET_COLUMNS.get(task)
    if target_col is None:
        raise KeyError(
            f"No default target column found for task '{task}'. "
            "Pass --target-col explicitly."
        )

    output_dir = resolve_output_dir(args, task=task, feature_set=feature_set)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_target_data(args, task=task)
    if target_col not in data.columns:
        raise KeyError(f"Target column '{target_col}' not found in input data.")

    fine_tune_df = select_fine_tuning_rows(
        data=data,
        target_col=target_col,
        split_by=args.split_by,
        n_mols=args.n_mols_fine_tune,
        random_state=args.fine_tune_random_state,
    )

    spec = TRANSFORMER_FEATURE_SPECS[feature_set]
    generator = FeatureGenerator(
        feature_set=feature_set,
        log_name=f"FT_diagnostics_{task}_{feature_set}",
    )

    result = generator.fineTuneTransformer(
        smiles_ls=fine_tune_df["SMILES"].to_list(),
        ids=fine_tune_df.index.to_list(),
        targets=fine_tune_df[target_col].astype(float).to_list(),
        model_label=spec["model_label"],
        batch_size=args.fine_tune_batch_size,
        max_token_len=spec["max_token_len"],
        pooling=args.pooling,
        output_dir=output_dir,
        test_frac=args.fine_tune_test_frac,
        early_stopping_patience=args.fine_tune_early_stopping_patience,
        training_kwargs={
            "num_train_epochs": args.fine_tune_epochs,
            "learning_rate": args.fine_tune_lr,
            "weight_decay": 0.01,
            "eval_strategy": "epoch",
            "save_strategy": "epoch",
            "logging_strategy": "steps",
            "logging_steps": 25,
            "load_best_model_at_end": True,
            "metric_for_best_model": "mse",
            "greater_is_better": False,
            "report_to": "none",
            "save_total_limit": 2,
            "save_safetensors": False,
        },
    )

    metrics = {
        "task": task,
        "feature_set": feature_set,
        "target_col": target_col,
        "n_selected_mols": len(fine_tune_df),
        "split_by": args.split_by,
        "fine_tune_epochs": args.fine_tune_epochs,
        "fine_tune_lr": args.fine_tune_lr,
        "fine_tune_batch_size": args.fine_tune_batch_size,
        "fine_tune_test_frac": args.fine_tune_test_frac,
        "early_stopping_patience": args.fine_tune_early_stopping_patience,
        "train_metrics": result["train_metrics"],
        "eval_metrics": result["eval_metrics"],
        "train_ids": result["train_ids"],
        "val_ids": result["val_ids"],
    }

    with (output_dir / "fine_tuning_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2, default=json_default)

    pd.DataFrame(
        [
            {
                "task": task,
                "feature_set": feature_set,
                "target_col": target_col,
                "n_selected_mols": len(fine_tune_df),
                "split_by": args.split_by,
                "fine_tune_epochs": args.fine_tune_epochs,
                "fine_tune_lr": args.fine_tune_lr,
                "fine_tune_batch_size": args.fine_tune_batch_size,
                "train_loss": result["train_metrics"].get("train_loss"),
                "eval_loss": result["eval_metrics"].get("eval_loss"),
                "eval_mse": result["eval_metrics"].get("eval_mse"),
                "eval_rmse": result["eval_metrics"].get("eval_rmse"),
                "eval_r2": result["eval_metrics"].get("eval_r2"),
            }
        ]
    ).to_csv(output_dir / "fine_tuning_metrics_summary.csv", index=False)

    if not args.keep_checkpoints:
        remove_checkpoints(output_dir)

    print(f"Saved fine-tuning diagnostics to: {output_dir}")
    print(f"Log history: {output_dir / 'fine_tuning_log_history.csv'}")
    print(f"Metrics: {output_dir / 'fine_tuning_metrics.json'}")


if __name__ == "__main__":
    main()
