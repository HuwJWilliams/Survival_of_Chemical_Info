# %%
from pathlib import Path
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
import logging
import joblib
from joblib import Parallel, delayed
import json
import torch

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[3]
DATASET_DIR = PROJ_DIR / "datasets"
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"
PALMERCHEM_SOFTWARE_ANALYSIS = Path.home() / "PalmerChem_Software" / "src" / "analysis"
LOG_DIR = SRC_DIR / "models" / "logs"

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData

sys.path.insert(0, str(SRC_DIR / "models"))
from RF_models import *
from extra_models import *

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import trimRowsByPercentile

# ========== Constants ========== #
BATCH_SIZE = 64
SEED = 42
LOG_LEVEL = logging.DEBUG


MODEL_REGISTRY = {
    "random_forest": {
        "regression": RFRegressor,
        "binary_classification": RFClassifier,
        "multiclass_classification": RFClassifier,
    },
    "ridge": {
        "regression": RidgeRegressor,
    },
    "svr": {"regression": SVRegressor},
}

from sklearn.model_selection import KFold, StratifiedKFold

TASK_CONFIG = {
    "regression": {
        "cv_function": KFold,
        "scoring": "neg_mean_squared_error",
    },
    "binary_classification": {
        "cv_function": StratifiedKFold,
        "scoring": "roc_auc",
    },
    "multiclass_classification": {
        "cv_function": StratifiedKFold,
        "scoring": "f1_macro",
    },
}


# ============ Class ============ #
class TL:
    def __init__(
        self,
        unembedded_df: pd.DataFrame = None,
        embedded_df: pd.DataFrame = None,
        seed: int = None,
        log_name: str = "TLModel",
        log_to_file: bool = False,
        log_dir: Path = LOG_DIR,
        log_level=LOG_LEVEL,
        log_identifier: str = "",
    ):
        """
        Initialise the transfer learning class

        Parameters
        ----------
        unembedded_df:  pd.DataFrame

        embedded_df:    pd.DataFrame

        seed:   int

        log_name:   str (optional)

        log_to_file:    bool (optional)

        log_dir:    Path, str (optional)

        log_level:

        log_identifier:     str (optional)


        """

        self.unembedded_df = unembedded_df
        self.embedded_df = embedded_df
        self.train_df = None
        self.test_df = None
        self.seed = seed
        self.instantiated_model = None

        if isinstance(log_dir, str):
            log_dir = Path(log_dir)

        # ===== Logger =====#
        self.logger = logging.getLogger(log_name)
        self.logger.setLevel(log_level)

        if not self.logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(funcName)s | %(message)s")
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            # File handler
            if log_to_file:
                log_dir.mkdir(parents=True, exist_ok=True)
                log_file = log_dir / f"{log_name}_{log_identifier}.log"
                file_handler = logging.FileHandler(log_file)
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)

        self.logger.info("Logger initialised.")

    def alignInputData(self, features: pd.DataFrame, targets: pd.DataFrame):

        common_indices = features.index.intersection(targets.index)
        if len(common_indices) == 0:
            error_msg = "No common indices found between features df and targets df"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        features = features.loc[common_indices]
        targets = targets.loc[common_indices]
        return features, targets

    def processInputData(
        self,
        target_column: str,
        target_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        remove_correlated_features: bool = False,
        trim_by_percentile: bool = True,
        percentile: float = 0.99,
        min_training_samples: int = 2500,
        min_samples_per_class: int = 20,
    ):
        """Prepare features and target, and identify the modelling task."""

        if target_column.upper() == "SMILES":
            self.logger.info(f"Skipping {target_column}: SMILES detected")
            return None

        if remove_correlated_features:
            raise NotImplementedError("Correlated-feature removal is not implemented.")

        # 1. Combine features and target
        features = feature_df.drop(
            columns=[target_column],
            errors="ignore",
        ).copy()

        combined_data = features.join(
            target_df[[target_column]],
            how="inner",
        )

        # 2. Clean invalid observations
        feature_columns = features.columns

        combined_data[feature_columns] = (
            combined_data[feature_columns]
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        valid_rows = (
            combined_data[feature_columns].notna().all(axis=1)
            & combined_data[target_column].notna()
        )

        n_removed = (~valid_rows).sum()
        combined_data = combined_data.loc[valid_rows].copy()

        if n_removed:
            self.logger.info(
                f"  {target_column}: removed {n_removed} rows "
                "with missing or non-finite values"
            )

        if combined_data.empty:
            self.logger.warning(f"Skipping {target_column}: no valid observations")
            return None

        # 3. Identify whether the target is categorical or numeric
        target = combined_data[target_column]

        is_categorical = pd.api.types.is_bool_dtype(target) or isinstance(
            target.dtype, pd.CategoricalDtype
        )

        if not is_categorical:
            numeric_target = pd.to_numeric(target, errors="coerce")

            if numeric_target.notna().all():
                combined_data[target_column] = numeric_target
                target = numeric_target
            else:
                is_categorical = True

        n_unique = target.nunique()

        if n_unique < 2:
            self.logger.warning(f"Skipping {target_column}: fewer than 2 unique values")
            return None

        # Numeric targets with 2-6 integer-like values are classified
        if not is_categorical:
            values = target.to_numpy(dtype=float)

            is_integer_like = np.all(np.isclose(values, np.round(values), atol=1e-9))

            is_categorical = is_integer_like and 2 <= n_unique <= 6

        # 4. Determine task and apply task-specific filtering
        if is_categorical:

            if n_unique > 6:
                self.logger.warning(
                    f"Skipping {target_column}: "
                    f"{n_unique} categorical classes exceed the limit of 6"
                )
                return None

            # Remove under-represented classes
            class_counts = combined_data[target_column].value_counts()

            valid_classes = class_counts[class_counts >= min_samples_per_class].index

            combined_data = combined_data.loc[
                combined_data[target_column].isin(valid_classes)
            ].copy()

            n_unique = combined_data[target_column].nunique()

            if n_unique < 2:
                self.logger.warning(
                    f"Skipping {target_column}: fewer than 2 classes "
                    "remain after filtering"
                )
                return None

            expected_task_type = (
                "binary_classification"
                if n_unique == 2
                else "multiclass_classification"
            )

        else:
            expected_task_type = "regression"

            # Trim extreme regression targets
            if trim_by_percentile:
                trimmed_target, _ = trimRowsByPercentile(
                    input_df=combined_data[[target_column]],
                    columns=[target_column],
                    percentile=percentile,
                    tail="upper",
                    return_removed_rows=True,
                )

                combined_data = combined_data.loc[
                    combined_data.index.intersection(trimmed_target.index)
                ].copy()

        # 5. Final validation
        n_samples = len(combined_data)

        if n_samples < min_training_samples:
            self.logger.warning(
                f"Skipping {target_column}: {n_samples} samples remain; "
                f"minimum is {min_training_samples}"
            )
            return None

        if combined_data[target_column].nunique() < 2:
            self.logger.warning(
                f"Skipping {target_column}: insufficient target variation"
            )
            return None

        self.logger.info(
            f"  Target: {target_column}\n"
            f"  Task: {expected_task_type}\n"
            f"  Samples: {n_samples}\n"
            f"  Unique target values: "
            f"{combined_data[target_column].nunique()}"
        )

        return combined_data, expected_task_type

    # --- Model Training
    def _train_one_target(
        self,
        target_column: str,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        existing_df: pd.DataFrame,
        model_name: str,
        hyper_params: dict | None,
        skip_existing: bool,
        n_resamples: int,
        test_size: float,
        cv_splits: int,
        batch_size: int,
        random_seed: int | None,
        save_models: bool,
        save_path: Path,
        trim_by_percentile: bool,
        percentile: float,
        min_training_samples: int,
        min_samples_per_class: int,
        outer_n_jobs: int,
        inner_n_jobs: int,
    ) -> dict:
        """Process and train one target. Returns a status dict for the parent.

        Runs in a worker process when target_n_jobs != 1, so it must not write to
        shared files. Log messages are returned rather than logged here, because
        loky workers don't inherit this class's logger handlers.
        """
        try:
            result = self.processInputData(
                target_column=target_column,
                target_df=targets_df,
                feature_df=features_df,
                remove_correlated_features=False,
                trim_by_percentile=trim_by_percentile,
                percentile=percentile,
                min_training_samples=min_training_samples,
                min_samples_per_class=min_samples_per_class,
            )
            if result is None:
                return {
                    "target": target_column,
                    "status": "skipped",
                    "message": f"Skipping {target_column}: failed data checks",
                }

            combined_df, expected_task_type = result

            if expected_task_type not in MODEL_REGISTRY[model_name]:
                return {
                    "target": target_column,
                    "status": "skipped",
                    "message": f"Skipping {target_column}: {model_name} "
                    f"does not support {expected_task_type}",
                }

            # Skip or rerun based on model and task type
            if (
                skip_existing
                and not existing_df.empty
                and {"model", "task_type"}.issubset(existing_df.columns)
            ):
                existing = existing_df.loc[
                    (existing_df.index == target_column)
                    & (existing_df["model"] == model_name)
                ]
                if (
                    not existing.empty
                    and (existing["task_type"] == expected_task_type).all()
                ):
                    return {
                        "target": target_column,
                        "status": "skipped",
                        "message": f"Skipping {model_name}: {target_column} "
                        f"(already processed as {expected_task_type})",
                    }

            model_class = MODEL_REGISTRY[model_name][expected_task_type]
            task_config = TASK_CONFIG[expected_task_type]

            model = model_class(
                cv_function=task_config["cv_function"],
                hp_search_function=GridSearchCV,
                cv_kwargs={"n_splits": cv_splits, "shuffle": True},
                hp_search_kwargs={"scoring": task_config["scoring"], "refit": True},
                random_seed=random_seed,
            )
            (
                _final_model,
                best_params,
                aggregated_performance,
                final_performance,
                feat_importance_df,
            ) = model.train(
                n_resamples=n_resamples,
                data=combined_df,
                target_column=target_column,
                hyperparameters=hyper_params,
                test_size=test_size,
                batch_size=batch_size,
                outer_n_jobs=outer_n_jobs,
                inner_n_jobs=inner_n_jobs,
                save_final_model=save_models,
                save_path=Path(save_path) / model_name / target_column,
                final_model_name=f"{target_column}_{model_name}",
            )
            return {
                "target": target_column,
                "status": "trained",
                "task_type": expected_task_type,
                "aggregated_performance": aggregated_performance,
                "best_params": best_params,
                "final_performance": final_performance,
                "feat_importance_df": feat_importance_df,
                "message": "",
            }

        except Exception as e:
            return {
                "target": target_column,
                "status": "error",
                "message": f"Error training model for {target_column}: {e}",
            }

    def trainMultiTargetModels(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        model_name: str = "random_forest",
        hyper_params: dict | None = None,
        output_csv: str = "total_performance.csv",
        existing_performance_csv: str = None,
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int | None = None,
        skip_existing: bool = True,
        save_models: bool = True,
        save_path: str = "./",
        trim_by_percentile: bool = True,
        percentile: float = 0.99,
        save_feat_imp: bool = False,
        min_training_samples: int = 2500,
        min_samples_per_class: int = 20,
        outer_n_jobs: int | None = None,
        inner_n_jobs: int | None = None,
        target_n_jobs: int = 1,
        recheck_task_type: bool = False,
    ) -> pd.DataFrame:
        """
        Train ML models for multiple target columns using logging.

        Classification targets with classes below min_samples_per_class are dropped
        rather than skipping the target entirely. If dropping small classes reduces
        the number of remaining classes to 2, the task falls back to binary
        classification. If fewer than 2 classes remain the target is skipped.

        Parallelism
        -----------
        target_n_jobs : int
            Number of targets trained concurrently (one worker per target). When
            != 1, each worker trains single-threaded (outer/inner n_jobs forced to
            1) to avoid oversubscription. When == 1, outer_n_jobs / inner_n_jobs
            are passed through to ModelBase.train as before.
        recheck_task_type : bool
            If False (default) and skip_existing is True, targets already present
            for this model are skipped up front, before any data processing. If
            True, the skip decision is made per target after task detection, so a
            changed task type triggers a re-run, at the cost of processing every
            target's data again on resumed runs.
        """

        total_performance_df = pd.DataFrame()

        outer_n_jobs = 1 if outer_n_jobs is None else outer_n_jobs
        inner_n_jobs = 1 if inner_n_jobs is None else inner_n_jobs

        if target_n_jobs != 1:
            # Parallelism is over targets; keep each worker single-threaded.
            outer_n_jobs, inner_n_jobs = 1, 1

        self.logger.debug(f"Existing performance CSV path: {existing_performance_csv}")

        # Load existing performance CSV
        if existing_performance_csv and Path(existing_performance_csv).exists():
            try:
                total_performance_df = loadData(existing_performance_csv, index_col=0)
                self.logger.info(
                    f"Loaded existing performance data from {existing_performance_csv}"
                )

            except Exception as e:
                self.logger.warning(f"Could not load existing performance CSV: {e}")

        # Align indices
        features_df, targets_df = self.alignInputData(
            features=features_df, targets=targets_df
        )

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        output_csv = save_path / output_csv
        combined_feat_importance_csv = save_path / "all_feature_importance.csv"
        best_params_csv = save_path / "best_params.csv"

        # Fast pre-skip: avoid dispatching (and processing data for) finished targets
        already_done = set()
        if (
            skip_existing
            and not recheck_task_type
            and not total_performance_df.empty
            and "model" in total_performance_df.columns
        ):
            already_done = set(
                total_performance_df.index[total_performance_df["model"] == model_name]
            )
            self.logger.info(
                f"Skipping {len(already_done)} targets already processed for {model_name}"
            )

        # Only pass the existing results to workers when they need them
        worker_existing_df = (
            total_performance_df if recheck_task_type else pd.DataFrame()
        )

        jobs = (
            delayed(self._train_one_target)(
                target_column=target_column,
                features_df=features_df,
                targets_df=targets_df,
                existing_df=worker_existing_df,
                model_name=model_name,
                hyper_params=hyper_params,
                skip_existing=skip_existing,
                n_resamples=n_resamples,
                test_size=test_size,
                cv_splits=cv_splits,
                batch_size=batch_size,
                random_seed=random_seed,
                save_models=save_models,
                save_path=save_path,
                trim_by_percentile=trim_by_percentile,
                percentile=percentile,
                min_training_samples=min_training_samples,
                min_samples_per_class=min_samples_per_class,
                outer_n_jobs=outer_n_jobs,
                inner_n_jobs=inner_n_jobs,
            )
            for target_column in targets_df.columns
            if target_column not in already_done
        )

        # Results are yielded as each target finishes, so the parent is the only
        # process that writes CSVs, and progress is saved after every target.
        for res in Parallel(n_jobs=target_n_jobs, return_as="generator_unordered")(
            jobs
        ):
            target_column = res["target"]

            if res["status"] == "error":
                self.logger.error(f"  {res['message']}")
                continue
            if res["status"] == "skipped":
                self.logger.info(res["message"])
                continue

            expected_task_type = res["task_type"]
            aggregated_performance = res["aggregated_performance"]
            final_performance = res["final_performance"]
            feat_importance_df = res["feat_importance_df"]

            best_params = res["best_params"]

            # Append best params (parent process is the only writer, so no races)
            bp_row = pd.DataFrame(
                [
                    {
                        "feature_name": target_column,
                        "best_params": json.dumps(best_params, default=str),
                    }
                ]
            )
            if best_params_csv.exists():
                # Drop any stale row for this target so re-runs don't duplicate it
                prev = pd.read_csv(best_params_csv)
                prev = prev.loc[prev["feature_name"] != target_column]
                bp_row = pd.concat([prev, bp_row], ignore_index=True)
            bp_row.to_csv(best_params_csv, index=False)

            #  Feature importance
            if (
                save_feat_imp
                and feat_importance_df is not None
                and not feat_importance_df.empty
            ):
                importance_col = f"Importance_{model_name}_{target_column}"

                fi_df = (
                    feat_importance_df[["Feature", "Importance"]]
                    .drop_duplicates(subset="Feature")
                    .set_index("Feature")
                    .rename(columns={"Importance": importance_col})
                )

                if combined_feat_importance_csv.exists():
                    saved_fi = pd.read_csv(
                        combined_feat_importance_csv,
                        index_col="Feature",
                    )

                    saved_fi = saved_fi.drop(
                        columns=[importance_col],
                        errors="ignore",
                    ).join(fi_df, how="outer")

                else:
                    saved_fi = fi_df

                saved_fi.to_csv(
                    combined_feat_importance_csv,
                    index_label="Feature",
                )

            elif save_feat_imp:
                self.logger.warning(
                    f"No feature importance for {model_name}: {target_column}"
                )

            #  Update performance df and save
            performance_row = {
                "model": model_name,
                "task_type": expected_task_type,
                **aggregated_performance,
                **{f"Final_{key}": value for key, value in final_performance.items()},
            }

            perf_df = pd.DataFrame(
                [performance_row],
                index=[target_column],
            )

            # Replace only the matching model-target result
            if (
                not total_performance_df.empty
                and "model" in total_performance_df.columns
            ):
                existing_mask = (total_performance_df.index == target_column) & (
                    total_performance_df["model"] == model_name
                )

                total_performance_df = total_performance_df.loc[~existing_mask]

            total_performance_df = pd.concat(
                [total_performance_df, perf_df],
                axis=0,
                sort=False,
            )

            total_performance_df.to_csv(output_csv)

            self.logger.info(
                f"Completed {model_name}: {target_column} "
                f"({expected_task_type}) → saved to {output_csv}"
            )

        self.logger.info(f"Completed training for {len(total_performance_df)} targets")
        self.logger.info(f"Results saved to: {output_csv}")

        return total_performance_df

    def trainWithinFeatureSetModels(
        self,
        data_df: pd.DataFrame,
        model_name: str,
        output_csv: str = "within_set_performance.csv",
        hyper_params: dict | None = None,
        save_path: str = "./",
        skip_existing: bool = True,
        save_models: bool = True,
        save_feat_imp: bool = False,
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 2,
        random_seed: int | None = None,
        trim_by_percentile: bool = True,
        percentile: float = 0.99,
        exclude_same_group: bool = False,
        group_map: dict | None = None,
    ):
        total_performance_df = pd.DataFrame()

        cv_splits = max(2, int(cv_splits))

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        output_csv = save_path / output_csv
        combined_feat_importance_csv = save_path / "all_feature_importance.csv"

        if output_csv.exists():
            try:
                prev = pd.read_csv(output_csv, index_col=0)
                total_performance_df = prev.copy()
            except Exception:
                pass

        data_df = data_df.copy().apply(pd.to_numeric, errors="coerce")

        # Optional group map
        desc_to_group = {}
        if exclude_same_group:
            if group_map is None:
                raise ValueError(
                    "exclude_same_group=True requires a valid group_map dictionary."
                )
            desc_to_group = {d: g for g, members in group_map.items() for d in members}

        for i, target_column in enumerate(data_df.columns):
            target_column = str(target_column).strip()

            y = data_df[target_column]

            # Start from all other descriptors
            feature_cols = [c for c in data_df.columns if c != target_column]

            # Remove same-group descriptors if requested
            if exclude_same_group:
                tgt_group = desc_to_group.get(target_column)
                if tgt_group is not None:
                    feature_cols = [
                        c for c in feature_cols if desc_to_group.get(c) != tgt_group
                    ]

            if len(feature_cols) == 0:
                self.logger.warning(
                    f"Skipping {target_column}: no predictors left after group exclusion"
                )
                continue

            # Process selected features and target
            result = self.processInputData(
                target_column=target_column,
                target_df=data_df[[target_column]],
                feature_df=data_df[feature_cols],
                trim_by_percentile=trim_by_percentile,
                percentile=percentile,
                min_training_samples=2500,
                min_samples_per_class=20,
            )

            if result is None:
                continue

            combined_data, expected_task_type = result

            if (
                skip_existing
                and not total_performance_df.empty
                and {"model", "task_type"}.issubset(total_performance_df.columns)
            ):
                existing = total_performance_df.loc[
                    (total_performance_df.index == target_column)
                    & (total_performance_df["model"] == model_name)
                ]

                if not existing.empty:
                    if (existing["task_type"] == expected_task_type).all():
                        self.logger.info(
                            f"Skipping {model_name}: {target_column} "
                            f"(already processed as {expected_task_type})"
                        )
                        continue

                    self.logger.info(
                        f"Re-running {model_name}: {target_column} "
                        f"because its task type changed to {expected_task_type}"
                    )

            # Select model and CV based on detected task
            model_class = MODEL_REGISTRY[model_name][expected_task_type]
            task_config = TASK_CONFIG[expected_task_type]
            try:
                self.instantiated_model = model_class(
                    cv_function=task_config["cv_function"],
                    hp_search_function=GridSearchCV,
                    cv_kwargs={
                        "n_splits": cv_splits,
                        "shuffle": True,
                    },
                    hp_search_kwargs={
                        "scoring": task_config["scoring"],
                        "refit": True,
                    },
                    random_seed=random_seed,
                )

                (
                    final_model,
                    best_params,
                    aggregated_performance,
                    final_performance,
                    feat_importance_df,
                ) = self.instantiated_model.train(
                    n_resamples=n_resamples,
                    data=combined_data,
                    target_column=target_column,
                    hyperparameters=hyper_params,
                    test_size=test_size,
                    save_interval_models=False,
                    save_path=save_path,
                    save_final_model=save_models,
                    plot_feat_importance=False,
                    outer_n_jobs=1,
                    inner_n_jobs=1,
                    final_model_name=f"{target_column}_{model_name}",
                )

                performance_row = {
                    "model": model_name,
                    "task_type": expected_task_type,
                    **aggregated_performance,
                    **{
                        f"Final_{key}": value
                        for key, value in final_performance.items()
                    },
                }

                perf_df = pd.DataFrame(
                    [performance_row],
                    index=[target_column],
                )

                # Replace only the matching model and target
                if (
                    not total_performance_df.empty
                    and "model" in total_performance_df.columns
                ):
                    existing_mask = (total_performance_df.index == target_column) & (
                        total_performance_df["model"] == model_name
                    )

                    total_performance_df = total_performance_df.loc[~existing_mask]

                total_performance_df = pd.concat(
                    [total_performance_df, perf_df],
                    axis=0,
                    sort=False,
                )
                total_performance_df.to_csv(output_csv)

            except Exception as e:
                self.logger.error(
                    f"Error training {model_name} for {target_column}: {e}"
                )
                continue

            if (
                save_feat_imp
                and feat_importance_df is not None
                and not feat_importance_df.empty
            ):
                importance_col = f"Importance_{model_name}_{target_column}"

                fi_df = (
                    feat_importance_df[["Feature", "Importance"]]
                    .drop_duplicates(subset="Feature")
                    .set_index("Feature")
                    .rename(columns={"Importance": importance_col})
                )

                if combined_feat_importance_csv.exists():
                    saved_fi = pd.read_csv(
                        combined_feat_importance_csv,
                        index_col="Feature",
                    )

                    saved_fi = saved_fi.drop(
                        columns=[importance_col],
                        errors="ignore",
                    ).join(fi_df, how="outer")

                else:
                    saved_fi = fi_df

                saved_fi.to_csv(
                    combined_feat_importance_csv,
                    index_label="Feature",
                )

            elif save_feat_imp:
                self.logger.info(
                    f"No feature importance available for {model_name}: {target_column}"
                )

        return total_performance_df

    def trainSingleTargetModel(
        self,
        data: pd.DataFrame,
        model_name: str,
        target_column: str,
        task_type: str = "regression",
        hyper_params: dict | None = None,
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int | None = None,
        save_models: bool = False,
        save_path: str = "./",
        log_level=logging.DEBUG,
        outer_n_jobs: int | None = None,
        inner_n_jobs: int | None = 1,
        trim_3xIQR: bool = True,
    ):

        # Initial setup defining/instantiating class and save paths
        model_class = MODEL_REGISTRY[model_name][task_type]
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        task_config = TASK_CONFIG[task_type]
        self.instantiated_model = model_class(
            cv_function=task_config["cv_function"],
            hp_search_function=GridSearchCV,
            cv_kwargs={"n_splits": cv_splits, "shuffle": True},
            hp_search_kwargs={"scoring": task_config["scoring"], "refit": True},
            random_seed=random_seed,
            log_level=log_level,
        )

        # Optionally trim the data to 3IQR
        if trim_3xIQR:
            values = data[target_column].dropna()

            q1 = values.quantile(0.25)
            q3 = values.quantile(0.75)
            iqr = q3 - q1

            lower_bound = q1 - (3.0 * iqr)
            upper_bound = q3 + (3.0 * iqr)

            outlier_mask = (data[target_column] < lower_bound) | (
                data[target_column] > upper_bound
            )

            trimmed_df = data.loc[~outlier_mask].copy()
            outliers_df = data.loc[outlier_mask].copy()

            n_total = data[target_column].notna().sum()
            n_outliers = outlier_mask.sum()
            n_remaining = trimmed_df[target_column].notna().sum()

            bounds = {
                "target_col": target_column,
                "q1": float(q1),
                "q3": float(q3),
                "iqr": float(iqr),
                "lower_bound": float(lower_bound),
                "upper_bound": float(upper_bound),
                "n_total": int(n_total),
                "n_outliers": int(n_outliers),
                "n_remaining": int(n_remaining),
                "percent_removed": (
                    float(100 * n_outliers / n_total) if n_total > 0 else 0.0
                ),
                "outliers": list(outliers_df.index.astype(str)),
            }

            bounds_path = save_path / f"{target_column}_3xIQR_bounds.json"

            with open(bounds_path, "w") as f:
                json.dump(bounds, f, indent=4)

            data = trimmed_df.copy()

            self.logger.info(
                f"3x IQR trimming applied to {target_column}: "
                f"removed {n_outliers}/{n_total} rows "
                f"({bounds['percent_removed']:.2f}%). "
                f"Bounds: {lower_bound:.4f} to {upper_bound:.4f}"
            )

        (
            final_model,
            best_params,
            aggregated_performance,
            final_performance,
            feat_importance_df,
        ) = self.instantiated_model.train(
            n_resamples=n_resamples,
            data=data,
            target_column=target_column,
            hyperparameters=hyper_params,
            test_size=test_size,
            save_interval_models=False,
            save_path=save_path,
            save_final_model=save_models,
            plot_feat_importance=False,
            batch_size=batch_size,
            outer_n_jobs=outer_n_jobs if outer_n_jobs is not None else 1,
            inner_n_jobs=inner_n_jobs,
            final_model_name=f"{target_column}_{model_name}",
        )

        if feat_importance_df is not None and not feat_importance_df.empty:
            feat_importance_df.to_csv(
                save_path / f"{target_column}_{model_name}_feature_importance.csv",
                index=False,
            )

        return (
            final_model,
            best_params,
            aggregated_performance,
            final_performance,
            feat_importance_df,
        )

    # --- Making Predictions
    def predictSingleTarget(
        self,
        model,
        model_name: str,
        data: pd.DataFrame | str | Path,
        task_type: str = "regression",
        target_column: str = "LD50",
        feature_cols: list[str] | None = None,
        calc_perf: bool = False,
        save_preds: bool = False,
        save_path: str | Path = Path(FILE_DIR),
        preds_filename: str = "preds",
        perf_filename: str = "performance",
    ):
        data = (
            loadData(data, index_col="ID")
            if isinstance(data, (str, Path))
            else data.copy()
        )

        if feature_cols is not None:
            feature_data = data[feature_cols].copy()
        else:
            feature_data = data.drop(
                columns=[target_column, "SMILES"], errors="ignore"
            ).copy()

        targets_true = (
            data[[target_column]].copy() if target_column in data.columns else None
        )

        if isinstance(model, (str, Path)):
            model = joblib.load(model)

        model_class = MODEL_REGISTRY[model_name][task_type]

        self.instantiated_model = model_class(
            cv_function=None,
            hp_search_function=None,
            cv_kwargs={},
            hp_search_kwargs={},
        )

        targets_pred = self.instantiated_model.predict(
            feature_data=feature_data,
            prediction_col=target_column,
            final_model=model,
            save_preds=save_preds,
            save_path=save_path,
            filename=preds_filename,
        )

        targ_pred = targets_pred[target_column]
        targ_true = targets_true[target_column] if targets_true is not None else None
        perf_dict = None

        if calc_perf:
            if targets_true is None:
                raise ValueError(
                    f"calc_perf=True but '{target_column}' is not present in the provided data."
                )

            perf_dict, _, _, _ = self.instantiated_model._calculate_performance(
                feature_test=feature_data,
                target_test=targets_true,
                best_model=model,
            )

            if save_preds:
                with open(Path(save_path) / f"{perf_filename}.json", "w") as f:
                    json.dump(perf_dict, f, indent=4)

        return targ_pred, targ_true, perf_dict

    # --- Miscellaneous Functions
    def rng(self) -> int:
        if self.seed is None:
            seed = random.randint(0, 2**32)
        else:
            seed = self.seed

        self.logger.info(f"Random seed: {seed}")
        return seed

    def _feature_reduction(
        self,
        feature_data,
        index_col: str = "ID",
        wildcard: str = "*",
        by_feature_correlation: bool = False,
        corr_threshold: float = 0.7,
    ) -> list[str]:

        feature_data = loadData(feature_data, index_col=index_col, wildcard=wildcard)

        if by_feature_correlation:
            # Get absolute correlation matric
            corr_matrix = feature_data.corr().abs()

            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )

            feats_to_drop = [
                column
                for column in upper.columns
                if any(upper[column] > corr_threshold)
            ]

        return feats_to_drop

    def _df_2_tensor(self, df: pd.DataFrame, dtype=torch.float32):
        return torch.tensor(df.to_numpy(), dtype=dtype)
