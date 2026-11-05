# %%
from pathlib import Path
import numpy as np
import pandas as pd
import random
import sys
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr
import logging
import joblib
import json
import torch
import os

FILE_DIR = Path(__file__).resolve()
PROJ_DIR = FILE_DIR.parents[3]
DATASET_DIR = PROJ_DIR / "datasets"
SCRIPTS_DIR = PROJ_DIR / "scripts"
SRC_DIR = SCRIPTS_DIR / "src"
PALMERCHEM_SOFTWARE_ANALYSIS = Path.home() / "PalmerChem_Software" / "src" / "analysis"
LOG_DIR = SRC_DIR / "models" / "logs"

sys.path.insert(0, str(SRC_DIR / "models"))
from Molecular_Embedding_vs_Descriptors.scripts.src.models.archive.RF_models import RFRegressor, RFClassifier, RFMultiClassifier

sys.path.insert(0, str(PALMERCHEM_SOFTWARE_ANALYSIS))
from performance_calculation import calculatePerformance

sys.path.insert(0, str(SRC_DIR / "misc"))
from misc_fns import loadData

sys.path.insert(0, str(SRC_DIR / "models"))
# from mlp_model import MLPRegressorTrainer, RegressionMLP

sys.path.insert(0, str(SRC_DIR / "datasets"))
from analyse_datasets import trimRowsByPercentile

# ========== Constants ========== #
BATCH_SIZE = 64
SEED = 42
LOG_LEVEL = logging.DEBUG


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

    # --- Model Training
    # --- --- Random Forests
    def trainMultiTargetRFModels(
        self,
        features_df: pd.DataFrame,
        targets_df: pd.DataFrame,
        rf_regressor_class=None,
        hyper_params: dict = {
            "n_estimators": [200, 400, 500],
            "max_features": ["sqrt"],
            "max_depth": [25, 50, 75, 100],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [2, 4, 8],
        },
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
        log_level=logging.DEBUG,
        trim_by_percentile: bool = True,
        percentile: float = 0.99,
        save_feat_imp: bool = False,
        min_training_samples: int = 2500,
        min_samples_per_class: int = 20,
        n_jobs: int | None = None,
    ) -> pd.DataFrame:
        """
        Train Random Forest models for multiple target columns using logging.

        Classification targets with classes below min_samples_per_class are dropped
        rather than skipping the target entirely. If dropping small classes reduces
        the number of remaining classes to 2, the task falls back to binary
        classification. If fewer than 2 classes remain the target is skipped.
        """


        total_performance_df = pd.DataFrame()
        regressor_cls = (
            rf_regressor_class if rf_regressor_class is not None else RFRegressor
        )
        completed_targets = set()

        # Cache n_jobs once so every target uses the same parallelism setting.
        n_jobs = n_jobs if n_jobs is not None else os.cpu_count()

        self.logger.debug(f"Existing performance CSV path: {existing_performance_csv}")

        # Load existing performance CSV
        if existing_performance_csv and Path(existing_performance_csv).exists():
            try:
                total_performance_df = loadData(existing_performance_csv, index_col=0)
                completed_targets = {str(c).strip() for c in total_performance_df.index}
                self.logger.info(
                    f"Loaded existing performance data from {existing_performance_csv}"
                )
            except Exception as e:
                self.logger.warning(f"Could not load existing performance CSV: {e}")

        # Align indices
        common_indices = features_df.index.intersection(targets_df.index)
        if len(common_indices) == 0:
            self.logger.error(
                "No common indices found between features_df and targets_df"
            )
            raise ValueError(
                "No common indices found between features_df and targets_df"
            )

        features_df = features_df.loc[common_indices]
        targets_df = targets_df.loc[common_indices]

        self.logger.info(
            f"Training models for {len(targets_df.columns)} target columns"
        )
        self.logger.info(
            f"Using {len(common_indices)} samples with {len(features_df.columns)} features"
        )

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        output_csv = save_path / output_csv
        combined_feat_importance_csv = save_path / "all_feature_importance.csv"

        for i, target_column in enumerate(targets_df.columns):
            target_column = str(target_column).strip()

            if target_column.upper() == "SMILES":
                self.logger.info(f"Skipping {target_column}: SMILES column detected")
                continue

            self.logger.info(
                f"Processing target: {target_column} ({i+1}/{len(targets_df.columns)})"
            )
            current_target = targets_df[target_column]
            current_features_df = features_df.drop(
                columns=[target_column], errors="ignore"
            )

            # Build combined data (inner join, no trimming yet)
            combined_data = pd.concat(
                [current_features_df, current_target], axis=1, join="inner"
            )

            # Drop rows where target is missing
            combined_data = combined_data.loc[
                combined_data[target_column].notna()
            ].copy()

            # ── Determine is_class_like BEFORE trimming ───────────────────────────
            non_na_target = combined_data[target_column].dropna()

            is_class_like = False
            if (
                pd.api.types.is_object_dtype(non_na_target)
                or isinstance(non_na_target.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(non_na_target)
            ):
                is_class_like = True
            elif pd.api.types.is_numeric_dtype(non_na_target):
                arr = non_na_target.to_numpy(dtype=float)
                is_class_like = bool(np.all(np.isclose(arr, np.round(arr), atol=1e-9)))

            # ── Trim by percentile only for regression targets ────────────────────
            if trim_by_percentile and not is_class_like:
                target_df = combined_data[[target_column]]
                trimmed_target_df, _ = trimRowsByPercentile(
                    input_df=target_df,
                    columns=[target_column],
                    percentile=percentile,
                    tail="upper",
                    return_removed_rows=True,
                )
                common_idx = current_features_df.index.intersection(
                    trimmed_target_df.index
                )
                combined_data = pd.concat(
                    [
                        current_features_df.loc[common_idx],
                        trimmed_target_df.loc[common_idx, target_column],
                    ],
                    axis=1,
                )
                combined_data = combined_data.loc[
                    combined_data[target_column].notna()
                ].copy()

            # ── Check minimum samples ─────────────────────────────────────────────
            if len(combined_data) < min_training_samples:
                self.logger.warning(
                    f"Skipping {target_column}: only {len(combined_data)} samples "
                    f"after filtering; minimum is {min_training_samples}"
                )
                continue

            # ── Check features are finite ─────────────────────────────────────────
            feature_cols = [c for c in combined_data.columns if c != target_column]
            inf_cols = [
                c
                for c in feature_cols
                if np.any(
                    ~np.isfinite(
                        combined_data[c].to_numpy(dtype=float, na_value=np.nan)
                    )
                )
            ]
            if inf_cols:
                self.logger.warning(
                    f"  {target_column}: {len(inf_cols)} feature column(s) contain NaN/inf values "
                    f"— dropping those rows. Columns: {inf_cols[:5]}{'...' if len(inf_cols) > 5 else ''}"
                )
                combined_data = combined_data.replace([np.inf, -np.inf], np.nan).dropna(
                    subset=feature_cols
                )

            if len(combined_data) < min_training_samples:
                self.logger.warning(
                    f"Skipping {target_column}: only {len(combined_data)} samples after dropping "
                    f"non-finite feature rows; minimum is {min_training_samples}"
                )
                continue

            # ── Re-derive non_na_target from cleaned data ─────────────────────────
            non_na_target = combined_data[target_column].dropna()
            n_unique = int(non_na_target.nunique())

            self.logger.info(f"  Samples after filtering: {len(combined_data)}")
            self.logger.info(f"  Target unique values: {n_unique}")

            if n_unique < 2:
                self.logger.warning(
                    f"  Skipping {target_column}: fewer than 2 unique target values"
                )
                continue

            # ── Cast numeric class-like targets to int ────────────────────────────
            if is_class_like and pd.api.types.is_numeric_dtype(non_na_target):
                combined_data[target_column] = (
                    pd.to_numeric(combined_data[target_column], errors="coerce")
                    .round()
                    .astype("Int64")
                )
                combined_data = combined_data.loc[
                    combined_data[target_column].notna()
                ].copy()
                combined_data[target_column] = combined_data[target_column].astype(int)

                # Recompute n_unique after casting
                n_unique = int(combined_data[target_column].nunique())

                if n_unique < 2:
                    self.logger.warning(
                        f"  Skipping {target_column}: fewer than 2 unique target values after casting"
                    )
                    continue

            # ── Assign initial task type ──────────────────────────────────────────
            if is_class_like and n_unique == 2:
                expected_task_type = "binary_classification"
            elif is_class_like and 3 <= n_unique <= 6:
                expected_task_type = "multiclass_classification"
            else:
                expected_task_type = "regression"

            # ── Filter under-represented classes ──────────────────────────────────
            # Done before the skip check so that expected_task_type is finalised
            # before any skip/re-run decision is made.
            if expected_task_type in {
                "binary_classification",
                "multiclass_classification",
            }:
                class_counts = combined_data[target_column].value_counts()
                small_classes = class_counts[
                    class_counts < min_samples_per_class
                ].index.tolist()

                if small_classes:
                    self.logger.warning(
                        f"  {target_column}: dropping {len(small_classes)} class(es) with "
                        f"fewer than {min_samples_per_class} samples: {small_classes}"
                    )
                    combined_data = combined_data[
                        ~combined_data[target_column].isin(small_classes)
                    ].copy()

                # Re-derive n_unique after dropping classes
                n_unique = int(combined_data[target_column].nunique())

                if n_unique < 2:
                    self.logger.warning(
                        f"Skipping {target_column}: fewer than 2 classes remain "
                        f"after dropping under-represented classes"
                    )
                    continue

                # Re-check minimum samples after row removal
                if len(combined_data) < min_training_samples:
                    self.logger.warning(
                        f"Skipping {target_column}: only {len(combined_data)} samples remain "
                        f"after dropping small classes; minimum is {min_training_samples}"
                    )
                    continue

                # Re-evaluate task type — only for classification targets.
                # Filtering can only reduce n_unique, so > 6 is unreachable here.
                if n_unique == 2:
                    expected_task_type = "binary_classification"
                elif 3 <= n_unique <= 6:
                    expected_task_type = "multiclass_classification"
                else:
                    raise RuntimeError(
                        f"Unexpected n_unique={n_unique} after class filtering for {target_column}; "
                        f"expected ≤ 6 since task was classification before filtering"
                    )

                self.logger.info(
                    f"  After class filtering: {n_unique} classes remain → {expected_task_type}"
                )

            # ── Skip or overwrite based on saved task type ────────────────────────
            if skip_existing and target_column in completed_targets:
                previous_task_type = None
                if "task_type" in total_performance_df.columns:
                    previous_task_type = total_performance_df.loc[
                        target_column, "task_type"
                    ]

                if previous_task_type == expected_task_type:
                    self.logger.info(
                        f"Skipping {target_column} ({i+1}/{len(targets_df.columns)})... "
                        f"already processed as {previous_task_type}"
                    )
                    continue

                self.logger.info(
                    f"Re-running {target_column} ({i+1}/{len(targets_df.columns)}): "
                    f"previous task_type={previous_task_type}, expected task_type={expected_task_type}"
                )

            self.logger.info(f"  n_unique: {n_unique}")
            self.logger.info(
                f"  unique values: {sorted(combined_data[target_column].dropna().unique())}"
            )
            self.logger.info(f"  is_class_like: {is_class_like}")
            self.logger.info(f"  expected_task_type: {expected_task_type}")

            # ── Train ─────────────────────────────────────────────────────────────
            if expected_task_type == "binary_classification":
                self.logger.info(f"  n_unique == 2 → RFClassifier")
                try:
                    rf_model = RFClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={
                            "n_splits": cv_splits,
                            "shuffle": True,
                            "random_state": random_seed,
                        },
                        hp_search_kwargs={"cv": cv_splits, "scoring": "roc_auc"},
                        log_level=log_level,
                        random_seed=random_seed,
                    )
                    final_model, best_params, performance_dict, feat_importance_df = (
                        rf_model.trainRFClassifier(
                            n_resamples=n_resamples,
                            data=combined_data,
                            target_column=target_column,
                            hyperparameters=hyper_params,
                            test_size=test_size,
                            save_interval_models=False,
                            save_path=save_path,
                            save_final_model=save_models,
                            plot_feat_importance=False,
                            batch_size=batch_size,
                            n_jobs=n_jobs,
                            final_rf_seed=None,
                            final_model_name=f"{target_column}_model",
                        )
                    )
                    performance_dict["task_type"] = "binary_classification"
                except Exception as e:
                    self.logger.error(
                        f"  Error training model for {target_column}: {str(e)}"
                    )
                    continue

            elif expected_task_type == "multiclass_classification":
                self.logger.info(f"  3 <= n_unique <= 6 → RFMultiClassifier")
                try:
                    rf_model = RFMultiClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={
                            "n_splits": cv_splits,
                            "shuffle": True,
                            "random_state": random_seed,
                        },
                        hp_search_kwargs={"cv": cv_splits, "scoring": "f1_macro"},
                        log_level=log_level,
                        random_seed=random_seed,
                    )
                    final_model, best_params, performance_dict, feat_importance_df = (
                        rf_model.trainRFMultiClassifier(
                            n_resamples=n_resamples,
                            data=combined_data,
                            target_column=target_column,
                            hyperparameters=hyper_params,
                            test_size=test_size,
                            save_interval_models=False,
                            save_path=save_path,
                            save_final_model=save_models,
                            plot_feat_importance=False,
                            batch_size=batch_size,
                            n_jobs=n_jobs,
                            final_rf_seed=None,
                            final_model_name=f"{target_column}_model",
                        )
                    )
                    performance_dict["task_type"] = "multiclass_classification"
                except Exception as e:
                    self.logger.error(
                        f"  Error training model for {target_column}: {str(e)}"
                    )
                    continue

            else:
                self.logger.info(f"  n_unique > 6 or not class-like → RFRegressor")
                try:
                    rf_model = regressor_cls(
                        cv_function=KFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={
                            "n_splits": cv_splits,
                            "shuffle": True,
                            "random_state": random_seed,
                        },
                        hp_search_kwargs={
                            "cv": cv_splits,
                            "scoring": "neg_mean_squared_error",
                        },
                        log_level=log_level,
                        random_seed=random_seed,
                    )
                    final_model, best_params, performance_dict, feat_importance_df = (
                        rf_model.trainRFRegressor(
                            n_resamples=n_resamples,
                            data=combined_data,
                            target_column=target_column,
                            hyperparameters=hyper_params,
                            test_size=test_size,
                            save_interval_models=False,
                            save_path=save_path,
                            save_final_model=save_models,
                            plot_feat_importance=False,
                            batch_size=batch_size,
                            n_jobs=n_jobs,
                            final_rf_seed=None,
                            final_model_name=f"{target_column}_model",
                        )
                    )
                    performance_dict["task_type"] = "regression"
                except Exception as e:
                    self.logger.error(
                        f"  Error training model for {target_column}: {str(e)}"
                    )
                    continue

            # ── Feature importance ────────────────────────────────────────────────
            if (
                save_feat_imp
                and feat_importance_df is not None
                and not feat_importance_df.empty
            ):
                safe_target_column = (
                    str(target_column).replace("/", "_").replace("\\", "_")
                )
                fi_df = feat_importance_df.copy()
                if "Feature" not in fi_df.columns:
                    fi_df = fi_df.reset_index().rename(columns={"index": "Feature"})

                if "Importance" in fi_df.columns:
                    imp_col = f"Importance_{safe_target_column}"
                    fi_df = fi_df[["Feature", "Importance"]].rename(
                        columns={"Importance": imp_col}
                    )
                    fi_df = fi_df.drop_duplicates(subset=["Feature"], keep="first")

                    if combined_feat_importance_csv.exists():
                        combined_df = pd.read_csv(combined_feat_importance_csv)
                        if "Feature" not in combined_df.columns:
                            combined_df = combined_df.reset_index().rename(
                                columns={"index": "Feature"}
                            )
                        combined_df = combined_df.drop_duplicates(
                            subset=["Feature"], keep="first"
                        )
                        stale_imp_cols = [
                            col
                            for col in combined_df.columns
                            if col == imp_col or col.startswith(f"{imp_col}_")
                        ]
                        combined_df = combined_df.loc[
                            :,
                            ["Feature"]
                            + [
                                col
                                for col in combined_df.columns
                                if col.startswith("Importance_")
                                and col not in stale_imp_cols
                            ],
                        ]
                        combined_df = combined_df.set_index("Feature")
                        fi_df = fi_df.set_index("Feature")
                        combined_df = combined_df.reindex(
                            combined_df.index.union(fi_df.index)
                        )
                        combined_df[imp_col] = fi_df[imp_col]
                        combined_df = combined_df.reset_index()
                    else:
                        combined_df = fi_df

                    combined_df.to_csv(combined_feat_importance_csv, index=False)
                else:
                    self.logger.warning(
                        f"  Feature importance for {target_column} has no 'Importance' column; "
                        "skipping cumulative feature-importance update."
                    )
            elif save_feat_imp:
                self.logger.warning(
                    f"  No feature importance dataframe returned for {target_column}"
                )

            # ── Update performance df and save ────────────────────────────────────
            perf_df = pd.DataFrame([performance_dict], index=[target_column])
            total_performance_df = total_performance_df.drop(
                index=[target_column], errors="ignore"
            )
            total_performance_df = pd.concat(
                [total_performance_df, perf_df], axis=0, sort=False
            )

            total_performance_df.to_csv(output_csv)

            self.logger.info(
                f"  Completed {target_column} ({performance_dict.get('task_type', 'unknown')}) → saved to {output_csv}"
            )

        self.logger.info(f"Completed training for {len(total_performance_df)} targets")
        self.logger.info(f"Results saved to: {output_csv}")

        return total_performance_df

    # region Hide code
    def trainWithinFeatureSetRFModels(
        self,
        data_df: pd.DataFrame,
        rf_regressor_class,
        output_csv: str = "within_set_performance.csv",
        hyper_params: dict = {
            "n_estimators": [200, 400, 500],
            "max_features": ["sqrt"],
            "max_depth": [25, 50, 75, 100],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [2, 4, 8],
        },
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
        completed_targets = set()
        cv_splits = max(2, int(cv_splits))

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        output_csv = save_path / output_csv
        combined_feat_importance_csv = save_path / "all_feature_importance.csv"

        if output_csv.exists():
            try:
                prev = pd.read_csv(output_csv, index_col=0)
                completed_targets = {str(i).strip() for i in prev.index}
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

            if skip_existing and target_column in completed_targets:
                self.logger.info(
                    f"Skipping {target_column} ({i+1}/{len(data_df.columns)})... already processed"
                )
                continue

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

            X = data_df[feature_cols]
            target_df = y.to_frame(name=target_column)

            if trim_by_percentile:
                trimmed_target_df, _removed_rows_df = trimRowsByPercentile(
                    input_df=target_df,
                    columns=[target_column],
                    percentile=percentile,
                    tail="upper",
                    return_removed_rows=True,
                )

                common_idx = X.index.intersection(trimmed_target_df.index)
                combined_data = pd.concat(
                    [
                        X.loc[common_idx],
                        trimmed_target_df.loc[common_idx, target_column],
                    ],
                    axis=1,
                )
            else:
                combined_data = pd.concat([X, y], axis=1, join="inner")

            # Pull out the target as a Series
            y = combined_data[target_column]
            # Drop rows where target is missing
            combined_data = combined_data.loc[
                combined_data[target_column].notna()
            ].copy()
            target_series = combined_data[target_column]
            n_unique = int(target_series.nunique(dropna=True))
            non_na_target = target_series.dropna()
            self.logger.info(
                f"  After removing non-finite values: {len(combined_data)} samples"
            )
            self.logger.info(f"  Target unique values: {n_unique}")

            if n_unique < 2:
                self.logger.warning(
                    f"  Skipping {target_column}: fewer than 2 unique target values"
                )
                continue

            # Class-like targets:
            # - object/category/bool labels
            # - numeric but all values are integer-like
            is_class_like = False
            if (
                pd.api.types.is_object_dtype(non_na_target)
                or pd.api.types.is_categorical_dtype(non_na_target)
                or pd.api.types.is_bool_dtype(non_na_target)
            ):
                is_class_like = True
            elif pd.api.types.is_numeric_dtype(non_na_target):
                arr = non_na_target.to_numpy(dtype=float)
                is_class_like = bool(np.all(np.isclose(arr, np.round(arr), atol=1e-9)))

            # Ensure sklearn classifiers see integer class labels for numeric class-like targets.
            if is_class_like and pd.api.types.is_numeric_dtype(non_na_target):
                combined_data[target_column] = (
                    pd.to_numeric(combined_data[target_column], errors="coerce")
                    .round()
                    .astype("Int64")
                )
                combined_data = combined_data.loc[
                    combined_data[target_column].notna()
                ].copy()
                combined_data[target_column] = combined_data[target_column].astype(int)

            try:
                if is_class_like and n_unique == 2:
                    rf_model = RFClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={
                            "n_splits": cv_splits,
                            "shuffle": True,
                            "random_state": random_seed,
                        },
                        hp_search_kwargs={"cv": cv_splits, "scoring": "roc_auc"},
                        random_seed=random_seed,
                    )

                    final_model, best_params, performance_dict, feat_importance_df = (
                        rf_model.trainRFClassifier(
                            n_resamples=n_resamples,
                            data=combined_data,
                            target_column=target_column,
                            test_size=test_size,
                            save_interval_models=False,
                            hyperparameters=hyper_params,
                            save_path=save_path,
                            save_final_model=save_models,
                            plot_feat_importance=False,
                            n_jobs=os.cpu_count(),
                            final_rf_seed=None,
                            final_model_name=f"{target_column}_model",
                        )
                    )
                    performance_dict["task_type"] = "binary_classification"

                elif is_class_like and n_unique <= 6:
                    rf_model = RFMultiClassifier(
                        cv_function=StratifiedKFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={
                            "n_splits": cv_splits,
                            "shuffle": True,
                            "random_state": random_seed,
                        },
                        hp_search_kwargs={"cv": cv_splits, "scoring": "f1_macro"},
                        random_seed=random_seed,
                    )

                    final_model, best_params, performance_dict, feat_importance_df = (
                        rf_model.trainRFMultiClassifier(
                            n_resamples=n_resamples,
                            data=combined_data,
                            target_column=target_column,
                            test_size=test_size,
                            save_interval_models=False,
                            hyperparameters=hyper_params,
                            save_path=save_path,
                            save_final_model=save_models,
                            plot_feat_importance=False,
                            n_jobs=os.cpu_count(),
                            final_rf_seed=None,
                            final_model_name=f"{target_column}_model",
                        )
                    )
                    performance_dict["task_type"] = "multiclass_classification"

                else:
                    rf_model = rf_regressor_class(
                        cv_function=KFold,
                        hp_search_function=GridSearchCV,
                        cv_kwargs={
                            "n_splits": cv_splits,
                            "shuffle": True,
                            "random_state": random_seed,
                        },
                        hp_search_kwargs={
                            "cv": cv_splits,
                            "scoring": "neg_mean_squared_error",
                        },
                        random_seed=random_seed,
                    )

                    final_model, best_params, performance_dict, feat_importance_df = (
                        rf_model.trainRFRegressor(
                            n_resamples=n_resamples,
                            data=combined_data,
                            target_column=target_column,
                            test_size=test_size,
                            save_interval_models=False,
                            hyperparameters=hyper_params,
                            save_path=save_path,
                            save_final_model=save_models,
                            plot_feat_importance=False,
                            n_jobs=os.cpu_count(),
                            final_rf_seed=None,
                            final_model_name=f"{target_column}_model",
                        )
                    )
                    performance_dict["task_type"] = "regression"

                perf_df = pd.DataFrame([performance_dict], index=[target_column])
                total_performance_df = total_performance_df.drop(
                    index=[target_column], errors="ignore"
                )
                total_performance_df = pd.concat(
                    [total_performance_df, perf_df], axis=0, sort=False
                )

                # Maintain one cumulative feature-importance CSV:
                # Feature, Importance_<target1>, Importance_<target2>, ...
                if (
                    save_feat_imp
                    and feat_importance_df is not None
                    and not feat_importance_df.empty
                ):
                    safe_target_column = (
                        str(target_column).replace("/", "_").replace("\\", "_")
                    )
                    fi_df = feat_importance_df.copy()
                    if "Feature" not in fi_df.columns:
                        fi_df = fi_df.reset_index().rename(columns={"index": "Feature"})

                    if "Importance" in fi_df.columns:
                        imp_col = f"Importance_{safe_target_column}"
                        fi_df = fi_df[["Feature", "Importance"]].rename(
                            columns={"Importance": imp_col}
                        )
                        fi_df = fi_df.drop_duplicates(subset=["Feature"], keep="first")

                        if combined_feat_importance_csv.exists():
                            combined_df = pd.read_csv(combined_feat_importance_csv)
                            if "Feature" not in combined_df.columns:
                                combined_df = combined_df.reset_index().rename(
                                    columns={"index": "Feature"}
                                )
                            combined_df = combined_df.drop_duplicates(
                                subset=["Feature"], keep="first"
                            )
                            stale_imp_cols = [
                                col
                                for col in combined_df.columns
                                if col == imp_col or col.startswith(f"{imp_col}_")
                            ]
                            combined_df = combined_df.loc[
                                :,
                                ["Feature"]
                                + [
                                    col
                                    for col in combined_df.columns
                                    if col.startswith("Importance_")
                                    and col not in stale_imp_cols
                                ],
                            ]
                            combined_df = combined_df.set_index("Feature")
                            fi_df = fi_df.set_index("Feature")
                            combined_df = combined_df.reindex(
                                combined_df.index.union(fi_df.index)
                            )
                            combined_df[imp_col] = fi_df[imp_col]
                            combined_df = combined_df.reset_index()
                        else:
                            combined_df = fi_df

                        combined_df.to_csv(combined_feat_importance_csv, index=False)
                    else:
                        self.logger.warning(
                            f"  Feature importance for {target_column} has no 'Importance' column; "
                            "skipping cumulative feature-importance update."
                        )
                elif save_feat_imp:
                    self.logger.warning(
                        f"  No feature importance dataframe returned for {target_column}"
                    )

                total_performance_df.to_csv(output_csv)
                self.logger.info(
                    f"Completed {target_column} ({performance_dict.get('task_type', 'unknown')})"
                )

            except Exception as e:
                self.logger.error(f"Error training model for {target_column}: {e}")
                continue

        return total_performance_df

    def trainSingleTargetRFModel(
        self,
        data: pd.DataFrame,
        target_column: str,
        hyper_params: dict = {
            "n_estimators": [400, 500],
            "max_features": ["sqrt"],
            "max_depth": [25, 50, 75, 100],
            "min_samples_split": [2, 5],
            "min_samples_leaf": [2, 4, 8],
        },
        n_resamples: int = 10,
        test_size: float = 0.3,
        cv_splits: int = 5,
        batch_size: int = 1,
        random_seed: int | None = None,
        save_models: bool = False,
        save_path: str = "./",
        log_level=logging.DEBUG,
        n_jobs: int | None = None,
        trim_3xIQR: bool = True,
    ):

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._rf_setup(
            random_seed=random_seed,
            cv_splits=cv_splits,
            log_level=log_level,
        )

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

        final_model, best_params, performance_dict, feat_importance_df = (
            self.instantiated_model.trainRFRegressor(
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
                n_jobs=n_jobs if n_jobs is not None else os.cpu_count(),
                final_rf_seed=None,
            )
        )

        joblib.dump(
            final_model,
            Path(save_path / f"{target_column}_RF_model.pkl.gz"),
            compress=("gzip", 3),
        )

        with open(Path(save_path / f"{target_column}_best_params.json"), "w") as f:
            json.dump(best_params, f, indent=4)

        with open(
            Path(save_path / f"{target_column}_internal_performance_dict.json"), "w"
        ) as f:
            json.dump(performance_dict, f, indent=4)

        feat_importance_df.to_csv(
            Path(save_path / f"{target_column}_feature_importance.csv")
        )

        return final_model, best_params, performance_dict, feat_importance_df

    # --- Making Predictions
    # --- --- Random Forests
    def predictSingleTargetRF(
        self,
        model,
        data: pd.DataFrame | str | Path,
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

        if not self.instantiated_model:
            self.instantiated_model = RFRegressor(
                cv_function=None,
                hp_search_function=None,
                cv_kwargs={},
                hp_search_kwargs={},
            )

        targets_pred = self.instantiated_model.predictRFRegressor(
            feature_data=feature_data,
            prediction_col=target_column,
            final_rf=model,
            save_preds=save_preds,
            save_path=save_path,
            filename=preds_filename,
        )

        targ_pred, targ_true, perf_dict = None, None, None

        if calc_perf:
            if targets_true is None:
                raise ValueError(
                    f"calc_perf=True but '{target_column}' is not present in the provided data."
                )

            targ_pred = (
                targets_pred[target_column]
                if isinstance(targets_pred, pd.DataFrame)
                and target_column in targets_pred.columns
                else targets_pred.squeeze()
            )
            targ_true = targets_true[target_column]

            perf_dict = self._calculate_performance(
                pred_targs=targ_pred, true_targs=targ_true
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

    def _rf_setup(
        self,
        random_seed=None,
        cv_splits: int = 5,
        log_level=logging.DEBUG,
    ):

        self.instantiated_model = RFRegressor(
            cv_function=KFold,
            hp_search_function=GridSearchCV,
            cv_kwargs={
                "n_splits": cv_splits,
                "shuffle": True,
                "random_state": random_seed,
            },
            hp_search_kwargs={"cv": cv_splits, "scoring": "neg_mean_squared_error"},
            log_level=log_level,
            random_seed=random_seed,
        )

        return self.instantiated_model

    def _calculate_performance(
        self,
        pred_targs,
        true_targs,
    ):
        """
        Returns
        -------
        Dictionary of performance metrics in this order-
        1. Bias
        2. Standard Error of Potential
        3. Mean Squared Error
        4. Root Mean Squared Error (computed from SDEP and Bias)
        5. Pearson R coefficient
        6. Spearman R coefficient
        7. r2 score
        """

        # Calculate Errors

        true_targs = np.asarray(true_targs, dtype=float)
        pred_targs = np.asarray(pred_targs, dtype=float)

        if true_targs.shape != pred_targs.shape:
            raise ValueError(
                f"Shape mismatch: true {true_targs.shape} vs pred {pred_targs.shape}"
            )

        errors = true_targs - pred_targs

        # Calculate performance metrics
        bias = np.mean(errors)
        sdep = (
            np.mean((true_targs - pred_targs - (np.mean(true_targs - pred_targs))) ** 2)
        ) ** 0.5
        mse = mean_squared_error(true_targs, pred_targs)
        rmse = np.sqrt(mse)
        r2 = r2_score(true_targs, pred_targs)

        # Pearson & Spearman correlations
        try:
            r_pearson, p_pearson = pearsonr(true_targs, pred_targs)
        except Exception as e:
            self.logger.warning(f"Pearson correlation failed: {e}\n")
            r_pearson, p_pearson = None, None

        try:
            r_spearman, _ = spearmanr(true_targs, pred_targs)
        except Exception as e:
            self.logger.warning(f"Spearman correlation failed: {e}\n")
            r_spearman = None

        return {
            "bias": bias,
            "sdep": sdep,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "r_pearson": r_pearson,
            "p_pearson": p_pearson,
            "r_spearman": r_spearman,
        }

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


# endregion
