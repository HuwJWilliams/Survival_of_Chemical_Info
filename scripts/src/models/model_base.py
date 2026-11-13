from __future__ import annotations

import inspect
import json
import logging as log
from pathlib import Path
from typing import Callable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def _json_default(value):
    """Convert NumPy scalar values to serializable built-in types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON-serialize {type(value).__name__}")


def _aggregate_metric_statistics(
    performance_list: list[dict], labels: dict[str, str]
) -> dict:
    """Summarise held-out metrics across resamples, ignoring non-finite scores."""
    means = {}
    spread = {}
    for metric, label in labels.items():
        values = np.asarray(
            [result[metric] for result in performance_list], dtype=float
        )
        values = values[np.isfinite(values)]

        means[label] = round(float(np.mean(values)), 4) if values.size else float("nan")

        spread[f"{label}_std"] = (
            round(float(np.std(values, ddof=1)), 4) if values.size > 1 else float("nan")
        )
        spread[f"{label}_min"] = (
            round(float(np.min(values)), 4) if values.size else float("nan")
        )
        spread[f"{label}_max"] = (
            round(float(np.max(values)), 4) if values.size else float("nan")
        )
        spread[f"{label}_range"] = (
            round(float(np.ptp(values)), 4) if values.size else float("nan")
        )

    return {**means, **spread}


class ModelBase:
    """Shared workflow for sklearn regressors, classifiers and pipelines.

    Parameters
    ----------
    cv_function : Callable
        CV splitter class, e.g. KFold or StratifiedKFold.
    hp_search_function : Callable
        Search class, e.g. GridSearchCV or RandomizedSearchCV.
    cv_kwargs, hp_search_kwargs : dict
        Configuration for the CV splitter and hyperparameter search.
    log_level : int
        Python logging level.
    random_seed : int or None
        Source seed for repeatable train/test and CV seeds.
    """

    def __init__(
        self,
        cv_function: Callable,
        hp_search_function: Callable,
        cv_kwargs: dict,
        hp_search_kwargs: dict,
        log_level: int = log.INFO,
        random_seed: int | None = None,
    ):
        self.logger = log.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        if not self.logger.hasHandlers():
            handler = log.StreamHandler()
            handler.setFormatter(log.Formatter("%(name)s - %(message)s"))
            self.logger.addHandler(handler)

        self.cv_function = cv_function
        self.cv_kwargs = dict(cv_kwargs)
        self.hp_search_function = hp_search_function
        self.hp_search_kwargs = dict(hp_search_kwargs)
        if "n_jobs" in self.hp_search_kwargs:
            raise ValueError(
                "Do not specify 'n_jobs' in hp_search_kwargs. "
                "Use 'inner_n_jobs' in train() instead."
            )
        if self.hp_search_kwargs.get("refit", True) is not True:
            raise ValueError(
                "Hyperparameter search must use refit=True "
                "because ModelBase requires best_estimator_."
            )

        self.random_seed = random_seed

    def _set_inner_cv(self, cv_kwargs: dict, cv_seed: int | None = None):
        """Construct the CV splitter; seed shuffling only when supported."""
        kwargs = dict(cv_kwargs)
        signature = inspect.signature(self.cv_function)
        if "random_state" in signature.parameters:
            if "shuffle" not in signature.parameters or kwargs.get("shuffle", False):
                kwargs["random_state"] = cv_seed
            else:
                # KFold / StratifiedKFold reject a random_state if shuffle=False.
                kwargs.pop("random_state", None)
        inner_cv = self.cv_function(**kwargs)
        return inner_cv, cv_seed

    def _set_hyperparameter_search(
        self,
        search_kwargs: dict,
        hyperparameters: dict,
        estimator,
        search_seed: int | None = None,
        inner_cv=None,
    ):
        """Construct GridSearchCV or RandomizedSearchCV with per-run CV."""
        kwargs = dict(search_kwargs)
        signature = inspect.signature(self.hp_search_function)
        if "random_state" in signature.parameters:
            kwargs["random_state"] = search_seed

        kwargs["estimator"] = estimator
        if inner_cv is None:
            raise ValueError("inner_cv must be provided for each resample")
        kwargs["cv"] = inner_cv
        if "param_grid" in signature.parameters:
            kwargs["param_grid"] = hyperparameters
        elif "param_distributions" in signature.parameters:
            kwargs["param_distributions"] = hyperparameters
        else:
            raise TypeError("Search class requires param_grid or param_distributions")
        return self.hp_search_function(**kwargs), search_seed

    def _prepare_data(
        self,
        data: pd.DataFrame | str | Path,
        target_column: str,
        metadata_columns: list | str | None = None,
        feature_columns: list | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:

        if isinstance(data, (str, Path)):
            data = pd.read_csv(data, index_col="ID")

        if isinstance(metadata_columns, str):
            metadata_columns = [metadata_columns]

        if feature_columns is not None:
            if target_column in feature_columns:
                raise ValueError("Target cannot be included in feature_columns")

            features = data.loc[:, feature_columns].copy()

        else:
            excluded = [target_column] + list(metadata_columns or [])
            features = data.drop(columns=excluded).copy()

        features = features.apply(pd.to_numeric, errors="coerce")
        targets = data[[target_column]]

        if features.shape[1] == 0:
            raise ValueError("No feature columns selected")

        if features.isna().any().any():
            invalid = features.columns[features.isna().any()].tolist()
            raise ValueError(f"Missing or non-numeric values in features: {invalid}")

        if targets[target_column].isna().any():
            raise ValueError("Training target contains missing values")

        return features, targets

    def _prepare_seed_lists(
        self,
        n_resamples: int,
        cv_seeds: list | None = None,
        search_seeds: list | None = None,
        split_seeds: list | None = None,
    ) -> tuple[list, list, list]:
        """Return per-resample CV, search and outer split seeds."""
        rng = np.random.default_rng(self.random_seed)
        if cv_seeds is None:
            cv_seeds = rng.integers(0, 2**31 - 1, n_resamples).tolist()
        if search_seeds is None:
            search_seeds = rng.integers(0, 2**31 - 1, n_resamples).tolist()
        if split_seeds is None:
            split_seeds = rng.integers(0, 2**31 - 1, n_resamples).tolist()
        if not all(
            len(values) == n_resamples
            for values in (cv_seeds, search_seeds, split_seeds)
        ):
            raise ValueError("Each seed list must have n_resamples entries")
        return cv_seeds, search_seeds, split_seeds

    def _resolve_final_seed(self, final_model_seed: int | None) -> int | None:
        """Use an explicitly selected final seed, otherwise the base seed."""
        return self.random_seed if final_model_seed is None else final_model_seed

    def _cast_best_params(self, best_params: dict) -> dict:
        """Convert NumPy scalar values to Python types, preserving None."""
        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in best_params.items()
        }

    def _plot_feature_importance(
        self,
        feat_importance_df: pd.DataFrame,
        top_n_feats: int = 20,
        save_data: bool = False,
        save_path: str | Path | None = None,
        filename: str = "feature_importance_plot",
        dpi: int = 300,
    ):
        """Plot and optionally save a model's feature importance DataFrame."""
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=feat_importance_df.head(top_n_feats),
            x="Importance",
            y="Feature",
            hue="Feature",
            dodge=False,
            legend=False,
            ax=ax,
        )
        ax.set(title="Feature Importances", xlabel="Importance", ylabel="Feature")
        if save_data:
            out = Path(save_path)
            out.mkdir(parents=True, exist_ok=True)
            fig.savefig(out / f"{filename}.png", dpi=dpi, bbox_inches="tight")
            feat_importance_df.to_csv(out / "feature_importance_df.csv", index=False)
        plt.close(fig)

    # ---- Hooks implemented or overridden by subclasses ----
    def _create_estimator(self):
        """Return a fresh unfitted estimator or sklearn Pipeline."""
        raise NotImplementedError("Subclasses must implement _create_estimator()")

    def _calculate_performance(self, feature_test, target_test, best_model):
        """Return (metrics, true, predicted, prediction_dataframe)."""
        raise NotImplementedError("Subclasses must implement _calculate_performance()")

    def _aggregate_performance(self, performance_list):
        """Aggregate per-resample metrics for the task."""
        raise NotImplementedError("Subclasses must implement _aggregate_performance()")

    def _get_feature_importance(self, model, feature_names):
        """Optionally return DataFrame with Feature and Importance columns."""
        return None

    def _get_uncertainty(self, model, features):
        """Optionally return per-row uncertainty, otherwise None."""
        return None

    def _get_probabilities(self, model, features):
        """Optionally return indexed probability DataFrame, otherwise None."""
        return None

    def _set_final_model_seed(self, model, seed):
        """Set random_state on an estimator or the last step of a pipeline."""
        if seed is None:
            return model
        params = model.get_params(deep=True)
        if "random_state" in params:
            model.set_params(random_state=seed)
        elif isinstance(model, Pipeline):
            last_name = model.steps[-1][0]
            param_name = f"{last_name}__random_state"
            if param_name in params:
                model.set_params(**{param_name: seed})
        return model

    def _split_data(self, features, targets, test_size, split_seed):
        """Regression's default outer split (classification overrides to stratify)."""
        return train_test_split(
            features,
            targets,
            test_size=test_size,
            random_state=split_seed,
        )

    # ---- Shared workflow ----
    def _fit_model_and_evaluate(
        self,
        resample_n: int,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        test_size: float,
        save_interval_models: bool,
        save_path: str | Path | None,
        hyperparameters: dict,
        cv_seed: int | None = None,
        split_seed: int | None = None,
        search_seed: int | None = None,
        inner_n_jobs: int = 1,
    ):
        """Fit/tune a fresh model on one outer split and evaluate it."""
        resample_number = resample_n + 1
        feat_tr, feat_te, tar_tr, tar_te = self._split_data(
            features,
            targets,
            test_size,
            split_seed,
        )
        tar_tr = np.asarray(tar_tr).ravel()
        tar_te = np.asarray(tar_te).ravel()
        inner_cv, _ = self._set_inner_cv(self.cv_kwargs, cv_seed)
        search, _ = self._set_hyperparameter_search(
            search_kwargs={
                **self.hp_search_kwargs,
                "n_jobs": inner_n_jobs,
            },
            hyperparameters=hyperparameters,
            estimator=self._create_estimator(),
            inner_cv=inner_cv,
            search_seed=search_seed,
        )
        search.fit(feat_tr, tar_tr)
        best_model = search.best_estimator_
        performance, true, pred, pred_df = self._calculate_performance(
            feature_test=feat_te,
            target_test=tar_te,
            best_model=best_model,
        )
        importance = self._get_feature_importance(best_model, features.columns)
        if save_interval_models:
            folder = Path(save_path)
            joblib.dump(best_model, folder / f"model_resample_{resample_number}.pkl")
            pred_df.to_csv(
                folder / f"preds_resample_{resample_number}.csv.gz",
                compression="gzip",
                index_label="ID",
            )
            with open(
                folder / f"performance_stats_resample_{resample_number}.json", "w"
            ) as handle:
                json.dump(performance, handle, indent=4, default=_json_default)
            if importance is not None:
                importance.to_csv(
                    folder / f"feature_importance_resample_{resample_number}.csv",
                    index=False,
                )
        return search.best_params_, performance, importance, resample_number, true, pred

    def train(
        self,
        n_resamples: int,
        data: pd.DataFrame | str | Path,
        target_column: str,
        hyperparameters: dict,
        test_size: float,
        metadata_columns: list | str | None = None,
        save_interval_models: bool = False,
        save_path: str | Path | None = None,
        save_final_model: bool = False,
        plot_feat_importance: bool = False,
        batch_size: int = 2,
        outer_n_jobs: int = 1,
        inner_n_jobs: int = 1,
        cv_seeds: list | None = None,
        search_seeds: list | None = None,
        split_seeds: list | None = None,
        final_model_seed: int | None = None,
        final_model_name: str = "final_model",
        save_training_ids_only: bool = True,
        feature_columns: list | None = None,
    ):
        """
        Run resampled evaluation and train a final model.

        Returns
        -------
        tuple
            final_model : fitted estimator
            best_params : dict
            aggregated_performance : dict
            final_training_performance : dict
            importance_df : pd.DataFrame or None
        """

        if n_resamples < 1 or batch_size < 1:
            raise ValueError("n_resamples and batch_size must be positive")
        if (
            save_interval_models or save_final_model or plot_feat_importance
        ) and save_path is None:
            raise ValueError("save_path is required when saving outputs")

        cv_seeds, search_seeds, split_seeds = self._prepare_seed_lists(
            n_resamples,
            cv_seeds,
            search_seeds,
            split_seeds,
        )
        final_model_seed = self._resolve_final_seed(final_model_seed)
        features, targets = self._prepare_data(
            data, target_column, metadata_columns, feature_columns=feature_columns
        )
        self._n_classes = (
            targets.iloc[:, 0].nunique() if isinstance(self, ClassifierBase) else None
        )
        output_path = Path(save_path) if save_path is not None else None
        self.interval_path = None
        if save_interval_models:
            self.interval_path = output_path / "all_resample_data"
            self.interval_path.mkdir(parents=True, exist_ok=True)

        def _process_batch(batch_indices):
            return [
                self._fit_model_and_evaluate(
                    resample_n=n,
                    features=features,
                    targets=targets,
                    test_size=test_size,
                    save_interval_models=save_interval_models,
                    save_path=self.interval_path,
                    hyperparameters=hyperparameters,
                    cv_seed=cv_seeds[n],
                    split_seed=split_seeds[n],
                    search_seed=search_seeds[n],
                    inner_n_jobs=inner_n_jobs,
                )
                for n in batch_indices
            ]

        batches = [
            range(i, min(i + batch_size, n_resamples))
            for i in range(0, n_resamples, batch_size)
        ]
        batch_results = Parallel(n_jobs=outer_n_jobs)(
            delayed(_process_batch)(batch) for batch in batches
        )
        results = [item for group in batch_results for item in group]
        (
            best_params_ls,
            self.performance_ls,
            importance_ls,
            self.resample_number_ls,
            self.true_vals_ls,
            self.pred_vals_ls,
        ) = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        self.performance_dict = self._aggregate_performance(self.performance_ls)

        importance_df = None
        if all(frame is not None for frame in importance_ls):
            series = [
                frame.set_index("Feature")["Importance"] for frame in importance_ls
            ]
            importance_df = (
                pd.concat(series, axis=1)
                .mean(axis=1)
                .rename("Importance")
                .reset_index()
                .sort_values("Importance", ascending=False)
            )
            if plot_feat_importance:
                self._plot_feature_importance(
                    importance_df,
                    save_data=True,
                    save_path=output_path,
                )
        elif plot_feat_importance:
            self.logger.warning("This model does not expose feature importance")

        # Perform final hyperparameter search on all training data
        final_cv, _ = self._set_inner_cv(
            cv_kwargs=self.cv_kwargs,
            cv_seed=final_model_seed,
        )

        final_estimator = self._set_final_model_seed(
            self._create_estimator(),
            final_model_seed,
        )

        final_search, _ = self._set_hyperparameter_search(
            search_kwargs={
                **self.hp_search_kwargs,
                "n_jobs": -1,
            },
            hyperparameters=hyperparameters,
            estimator=final_estimator,
            search_seed=final_model_seed,
            inner_cv=final_cv,
        )

        final_search.fit(features, targets.to_numpy().ravel())

        best_params = self._cast_best_params(final_search.best_params_)
        self.final_model = final_search.best_estimator_

        # Evaluate final fitted model on the full training dataset
        (
            self.final_training_performance,
            final_true,
            final_pred,
            final_pred_df,
        ) = self._calculate_performance(
            feature_test=features,
            target_test=targets,
            best_model=self.final_model,
        )

        if save_final_model:
            training_path = output_path / "training_data"
            training_path.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                self.final_model,
                training_path / f"{final_model_name}.pkl.gz",
                compress=("gzip", 3),
            )

            with open(training_path / "performance_stats.json", "w") as handle:
                json.dump(
                    self.performance_dict, handle, indent=4, default=_json_default
                )

            with open(training_path / "final_training_performance.json", "w") as handle:
                json.dump(
                    self.final_training_performance,
                    handle,
                    indent=4,
                    default=_json_default,
                )

            # Save final training predictions and true values
            final_pred_df = final_pred_df.copy()
            final_pred_df["True"] = final_true

            final_pred_df.to_csv(
                training_path / "final_training_predictions.csv.gz",
                compression="gzip",
                index_label="ID",
            )

            with open(training_path / "best_params.json", "w") as handle:
                json.dump(best_params, handle, indent=4, default=_json_default)
            if save_training_ids_only:
                targets.index.to_frame(index=False, name="ID").to_csv(
                    training_path / "training_ids.csv",
                    index=False,
                )
            else:
                features.to_csv(
                    training_path / "training_features.csv.gz",
                    compression="gzip",
                    index_label="ID",
                )
                targets.to_csv(
                    training_path / "training_targets.csv.gz",
                    compression="gzip",
                    index_label="ID",
                )
        self.logger.info("Final %s trained", type(self.final_model).__name__)
        return (
            self.final_model,
            best_params,
            self.performance_dict,
            self.final_training_performance,
            importance_df,
        )

    def predict(
        self,
        feature_data: pd.DataFrame | str | Path,
        prediction_col: str = "Prediction",
        final_model=None,
        save_preds: bool = False,
        save_path: str | Path | None = None,
        filename: str | None = None,
    ) -> pd.DataFrame:
        """Predict with a fitted model or saved model, retaining ID index.

        Subclass hooks optionally append class probability and uncertainty columns.
        """
        if isinstance(feature_data, (str, Path)):
            feature_data = pd.read_csv(feature_data, index_col="ID")

        if isinstance(final_model, (str, Path)):
            model = joblib.load(final_model)

        elif final_model is not None:
            model = final_model

        else:
            model = getattr(self, "final_model", None)

            if model is None:
                raise ValueError(
                    "No fitted model available. "
                    "Call train() first or provide a fitted model "
                    "using final_model."
                )

        feature_data = feature_data.apply(pd.to_numeric, errors="coerce")
        if hasattr(model, "feature_names_in_"):
            feature_data = feature_data.reindex(columns=model.feature_names_in_)
        if feature_data.isna().any().any():
            raise ValueError(
                "Prediction features have missing, nonnumeric, or required missing columns"
            )
        if (
            hasattr(model, "n_features_in_")
            and feature_data.shape[1] != model.n_features_in_
        ):
            raise ValueError("Prediction feature count differs from fitted model")

        predictions = np.asarray(model.predict(feature_data)).ravel()
        result = pd.DataFrame({prediction_col: predictions}, index=feature_data.index)
        probabilities = self._get_probabilities(model, feature_data)
        if probabilities is not None:
            # Assign positionally: joining on duplicate IDs would multiply rows.
            for column in probabilities.columns:
                result[column] = probabilities[column].to_numpy()
        uncertainty = self._get_uncertainty(model, feature_data)
        if uncertainty is not None:
            result["Uncertainty"] = uncertainty

        if save_preds:
            if save_path is None or filename is None:
                raise ValueError(
                    "Both save_path and filename are required when saving predictions"
                )
            folder = Path(save_path)
            folder.mkdir(parents=True, exist_ok=True)
            result.to_csv(
                folder / f"{filename}.csv.gz", compression="gzip", index_label="ID"
            )
        return result


class RegressorBase(ModelBase):
    """Shared regression performance metrics and aggregation."""

    def _calculate_performance(self, feature_test, target_test, best_model):
        true = np.asarray(target_test, dtype=float).ravel()
        pred = np.asarray(best_model.predict(feature_test), dtype=float).ravel()
        error = true - pred
        mse = mean_squared_error(true, pred)
        try:
            pearson, p_value = pearsonr(true, pred)
        except (ValueError, TypeError):
            pearson, p_value = np.nan, np.nan
        try:
            spearman, _ = spearmanr(true, pred)
        except (ValueError, TypeError):
            spearman = np.nan
        metrics = {
            "bias": float(np.mean(error)),
            "sdep": float(np.std(error)),
            "mse": float(mse),
            "rmse": float(np.sqrt(mse)),
            "r2": float(r2_score(true, pred)),
            "r_pearson": float(pearson),
            "p_pearson": float(p_value),
            "r_spearman": float(spearman),
        }
        pred_df = pd.DataFrame({"Prediction": pred}, index=feature_test.index)
        return metrics, true, pred, pred_df

    def _aggregate_performance(self, performance_list):

        labels = {
            "bias": "Bias",
            "sdep": "SDEP",
            "mse": "MSE",
            "rmse": "RMSE",
            "r2": "r2",
            "r_pearson": "Pearson_r",
            "r_spearman": "Spearman_r",
        }

        return _aggregate_metric_statistics(performance_list, labels)


class ClassifierBase(ModelBase):
    """Shared binary/multiclass classification metrics and probability formatting."""

    def _split_data(self, features, targets, test_size, split_seed):
        labels = np.asarray(targets).ravel()

        classes, counts = np.unique(labels, return_counts=True)

        if counts.min() < 2:
            raise ValueError(
                "Stratified splitting requires at least 2 samples "
                "per class. Class counts: "
                f"{dict(zip(classes.tolist(), counts.tolist()))}"
            )

        return train_test_split(
            features,
            targets,
            test_size=test_size,
            random_state=split_seed,
            stratify=labels,
        )

    def _get_probabilities(self, model, features):
        if not callable(getattr(model, "predict_proba", None)):
            return None
        probs = np.asarray(model.predict_proba(features))
        classes = np.asarray(model.classes_)
        if probs.ndim != 2 or probs.shape[1] != len(classes):
            raise ValueError("Estimator returned unexpected probability shape")
        if len(classes) == 2:
            return pd.DataFrame({"Pred_Prob": probs[:, 1]}, index=features.index)
        columns = {
            f"Pred_Prob_{label}": probs[:, idx] for idx, label in enumerate(classes)
        }
        result = pd.DataFrame(columns, index=features.index)
        result["Pred_Prob_Max"] = probs.max(axis=1)
        return result

    def _calculate_performance(self, feature_test, target_test, best_model):
        true = np.asarray(target_test).ravel()
        pred = np.asarray(best_model.predict(feature_test)).ravel()
        classes = np.asarray(best_model.classes_)
        metrics = {
            "acc": float(accuracy_score(true, pred)),
            "bal_acc": float(balanced_accuracy_score(true, pred)),
            "mcc": float(matthews_corrcoef(true, pred)),
            "f1_macro": float(f1_score(true, pred, average="macro", zero_division=0)),
            "sens": np.nan,
            "spec": np.nan,
            "ppv": np.nan,
            "npv": np.nan,
            "auc": np.nan,
        }
        probability_df = self._get_probabilities(best_model, feature_test)
        if len(classes) == 2:
            tn, fp, fn, tp = confusion_matrix(true, pred, labels=classes).ravel()
            metrics["sens"] = float(tp / (tp + fn)) if tp + fn else np.nan
            metrics["spec"] = float(tn / (tn + fp)) if tn + fp else np.nan
            metrics["ppv"] = float(tp / (tp + fp)) if tp + fp else np.nan
            metrics["npv"] = float(tn / (tn + fn)) if tn + fn else np.nan
        if probability_df is not None:
            try:
                if len(classes) == 2:
                    metrics["auc"] = float(
                        roc_auc_score(true, probability_df["Pred_Prob"])
                    )
                else:
                    scores = np.asarray(best_model.predict_proba(feature_test))
                    metrics["auc"] = float(
                        roc_auc_score(
                            true,
                            scores,
                            labels=classes,
                            multi_class="ovr",
                        )
                    )
            except ValueError:
                metrics["auc"] = np.nan
        pred_df = pd.DataFrame({"Prediction": pred}, index=feature_test.index)
        if probability_df is not None:
            for column in probability_df.columns:
                pred_df[column] = probability_df[column].to_numpy()
        return metrics, true, pred, pred_df

    def _aggregate_performance(self, performance_list):
        if self._n_classes is not None and self._n_classes > 2:
            labels = {
                "acc": "Accuracy",
                "bal_acc": "Balanced_Accuracy",
                "f1_macro": "F1_macro",
                "auc": "AUC_OVR",
                "mcc": "MCC",
            }
        else:
            labels = {
                "acc": "Accuracy",
                "bal_acc": "Balanced_Accuracy",
                "sens": "Sensitivity",
                "spec": "Specificity",
                "ppv": "PPV",
                "npv": "NPV",
                "auc": "AUC",
                "mcc": "MCC",
            }
        return _aggregate_metric_statistics(performance_list, labels)
