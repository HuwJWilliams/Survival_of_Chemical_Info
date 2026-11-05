import logging as log
import inspect
from typing import Callable
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json

from sklearn.model_selection import train_test_split
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


class ModelBase:
    """
    Base class providing shared utilities for machine learning pipelines.

    Provides methods for data preparation, cross-validation configuration,
    hyperparameter search, random seed management, parameter selection,
    and feature importance visualization.

    This class is intended to be inherited by model-specific classes
    implementing training, evaluation, and prediction workflows.

    Parameters
    ----------
    cv_function : Callable
        Cross-validation splitter class, such as KFold or StratifiedKFold.
    hp_search_function : Callable
        Hyperparameter search class, such as GridSearchCV.
    cv_kwargs : dict
        Keyword arguments used to initialize the CV splitter.
    hp_search_kwargs : dict
        Keyword arguments used to initialize the hyperparameter search.
    log_level : int, default=logging.INFO
        Logging level for the model.
    random_seed : int or None, default=None
        Base random seed used to generate reproducible seeds for
        cross-validation, hyperparameter search, and data splitting.
    """

    def __init__(
        self,
        cv_function: Callable,
        hp_search_function: Callable,
        cv_kwargs: dict,
        hp_search_kwargs: dict,
        log_level: int = log.INFO,
        random_seed: int = None,
    ):

        # Logge
        self.logger = log.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)

        if not self.logger.hasHandlers():
            handler = log.StreamHandler()
            formatter = log.Formatter("%(name)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

        self.cv_function = cv_function
        self.cv_kwargs = cv_kwargs
        self.hp_search_function = hp_search_function
        self.hp_search_kwargs = hp_search_kwargs

        self.random_seed = random_seed

    def _set_inner_cv(self, cv_kwargs: dict, cv_seed: int = None):
        """
        Initialize the inner cross-validation splitter.

        Parameters
        ----------
        cv_kwargs : dict
            Keyword arguments passed to the cross-validation splitter.
        cv_seed : int or None, default=None
            Random seed passed to the splitter if it supports random_state.

        Returns
        -------
        tuple
            Initialized cross-validation splitter and the CV seed.
        """
        kwargs = dict(cv_kwargs)
        init_sig = inspect.signature(self.cv_function)

        # Only pass random_state to splitters that support it.
        if "random_state" in init_sig.parameters:
            kwargs["random_state"] = cv_seed
            self.logger.debug(
                f"Set random_state={cv_seed} in cv_kwargs for {self.cv_function.__name__}\n"
            )

        self.inner_cv = self.cv_function(**kwargs)
        return self.inner_cv, cv_seed

    def _set_hyperparameter_search(
        self,
        search_kwargs: dict,
        hyperparameters: dict,
        estimator,
        search_seed: int = None,
    ):
        """
        Initialize a hyperparameter search using the configured estimator.

        Parameters
        ----------
        search_kwargs : dict
            Keyword arguments passed to the search class.
        hyperparameters : dict
            Hyperparameter grid to evaluate.
        estimator : sklearn estimator
            Estimator to optimize.
        search_seed : int or None, default=None
            Random seed for the search algorithm, if supported.

        Returns
        -------
        tuple
            Initialized hyperparameter search object and search seed.

        Notes
        -----
        Uses the inner CV splitter previously initialized by
        _set_inner_cv().

        The current implementation supports grid-based searches
        using the param_grid argument.
        """

        kwargs = dict(search_kwargs)
        init_sig = inspect.signature(self.hp_search_function)
        if "random_state" in init_sig.parameters:
            kwargs["random_state"] = search_seed
            self.logger.debug(
                f"Set random_state={search_seed} in search_kwargs for {self.hp_search_function.__name__}\n"
            )

        kwargs["param_grid"] = hyperparameters
        kwargs["estimator"] = estimator
        kwargs["cv"] = self.inner_cv
        hp_search_object = self.hp_search_function(**kwargs)
        return hp_search_object, search_seed

    def _prepare_data(
        self,
        data: pd.DataFrame | str,
        target_column: str,
        feature_columns: list[str],
        metadata_columns: list | str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare feature and target DataFrames for model training.

        Parameters
        ----------
        data : pd.DataFrame or str
            Training dataset or path to a CSV file.
            CSV files must contain an 'ID' column.
        target_column : str
            Name of the column containing the prediction target.
        metadata_columns : list, str, or None, default=None
            Columns to exclude from the feature matrix.

        Returns
        -------
        """

        if isinstance(data, str):
            data = pd.read_csv(data, index_col="ID")

        if isinstance(metadata_columns, str):
            metadata_columns = [metadata_columns]

        columns_to_drop = [target_column]
        if metadata_columns:
            columns_to_drop = list(metadata_columns) + columns_to_drop

        if feature_columns is not None:
            features = data.loc[:, feature_columns].copy()
        else:
            excluded = [target_column] + list(metadata_columns or [])
            features = data.drop(columns=excluded).copy()

        features = features.apply(pd.to_numeric, errors="coerce")

        if features.isna().any().any():
            invalid = features.columns[features.isna().any()].tolist()
            raise ValueError(f"Missing or non-numeric values in features: {invalid}")

        targets = data[[target_column]]
        return features, targets

    def _prepare_seed_lists(
        self,
        n_resamples: int,
        cv_seeds: list | None,
        split_seeds: list | None,
        search_seeds: list | None,
    ) -> tuple[list, list]:
        """
        Prepare random seed lists for repeated model evaluation.

        Generates seeds for cross-validation, hyperparameter search,
        and train/test splitting when explicit lists are not supplied.

        Parameters
        ----------
        n_resamples : int
            Number of repeated model evaluations.
        cv_seeds : list or None, default=None
            Seeds used to initialize cross-validation splitters.
        search_seeds : list or None, default=None
            Seeds controlling hyperparameter search randomness.
        split_seeds : list or None, default=None
            Seeds controlling outer train/test splits.

        Returns
        -------
        cv_seeds : list
            Cross-validation seeds.
        search_seeds : list
            Hyperparameter search seeds.
        split_seeds : list
            Train/test split seeds.

        Notes
        -----
        Missing seed lists are generated using a NumPy random
        generator initialized with self.random_seed.

        When self.random_seed is fixed, generated seed lists
        are reproducible.
        """

        # Use one generator for all three seed lists.
        rng = np.random.default_rng(self.random_seed)

        if cv_seeds is None:
            cv_seeds = rng.integers(0, 2**31 - 1, size=n_resamples).tolist()

        if search_seeds is None:
            search_seeds = rng.integers(0, 2**31 - 1, size=n_resamples).tolist()

        if split_seeds is None:
            split_seeds = rng.integers(0, 2**31 - 1, size=n_resamples).tolist()

        assert len(cv_seeds) == n_resamples
        assert len(search_seeds) == n_resamples
        assert len(split_seeds) == n_resamples

        return cv_seeds, search_seeds, split_seeds

    def _resolve_final_seed(self, final_rf_seed: int | None) -> int:
        if final_rf_seed is not None:
            return final_rf_seed
        return self.random_seed

    def _cast_best_params(self, best_params: dict) -> dict:
        """Convert NumPy scalar values to Python types, preserving None."""
        return {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in best_params.items()
        }

    def _derive_best_params(self, best_params_ls: list[dict]) -> dict:
        """
        Selects the most frequently occurring value of hyper parameter across resamples
        """

        best_params_df = pd.DataFrame(best_params_ls)
        derived = {}
        for col in best_params_df.columns:
            non_null = best_params_df[col].dropna()
            if non_null.empty:
                continue
            modes = non_null.mode()
            derived[col] = modes.iloc[0] if not modes.empty else non_null.iloc[0]
        return derived

    def _plot_feature_importance(
        self,
        feat_importance_df: pd.DataFrame = None,
        top_n_feats: int = 20,
        save_data: bool = False,
        save_path: str = None,
        filename: str = None,
        dpi: int = 500,
    ):
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=feat_importance_df.head(top_n_feats),
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

        if save_data:
            plt.savefig(f"{save_path}/{filename}.png", dpi=dpi)
            feat_importance_df.to_csv(f"{save_path}/feature_importance_df.csv")
        return

    def _create_estimator(self):
        raise NotImplementedError

    def _calculate_performance(self, y_true, y_pred, **kwargs):
        raise NotImplementedError

    def _get_feature_importance(self, model, feature_names):
        return None

    def _get_uncertainty(self, model, features):
        return None


def _fit_model_and_evaluate(
    self,
    resample_n: int,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    test_size: float,
    save_interval_models: bool,
    save_path: str,
    hyperparameters: dict,
    cv_seed: int = None,
    split_seed: int = None,
    search_seed: int = None,
):
    resample_n += 1

    # Split data into training and test sets
    feat_tr, feat_te, tar_tr, tar_te = train_test_split(
        features,
        targets,
        test_size=test_size,
        random_state=split_seed,
    )

    tar_tr = np.asarray(tar_tr).ravel()
    tar_te = np.asarray(tar_te).ravel()

    # Create model-specific estimator
    model_estimator = self._create_estimator()

    # Initialize inner cross-validation
    inner_cv, _ = self._set_inner_cv(
        cv_kwargs=self.cv_kwargs,
        cv_seed=cv_seed,
    )

    # Initialize hyperparameter search
    model, _ = self._set_hyperparameter_search(
        search_kwargs={**self.hp_search_kwargs, "cv": inner_cv},
        hyperparameters=hyperparameters,
        estimator=model_estimator,
        search_seed=search_seed,
    )

    model.fit(feat_tr, tar_tr)
    best_model = model.best_estimator_

    # Calculate regression or classification performance
    performance, true, pred, pred_df = self._calculate_performance(
        feature_test=feat_te,
        target_test=tar_te,
        best_model=best_model,
    )

    # Optional model-specific feature importance
    feat_importance_df = self._get_feature_importance(
        best_model,
        features.columns,
    )

    if save_interval_models:
        joblib.dump(
            best_model,
            f"{save_path}/model_resample_{resample_n}.pkl",
        )

        pred_df.to_csv(
            f"{save_path}/preds_resample_{resample_n}.csv.gz",
            compression="gzip",
            index_label="ID",
        )

        with open(
            f"{save_path}/performance_stats_resample_{resample_n}.json",
            "w",
        ) as file:
            json.dump(performance, file, indent=4)

        if feat_importance_df is not None:
            feat_importance_df.to_csv(
                f"{save_path}/feature_importance_resample_{resample_n}.csv",
                index=False,
            )

    return (
        model.best_params_,
        performance,
        feat_importance_df,
        resample_n,
        true,
        pred,
    )


def train(
    self,
    n_resamples: int,
    data: pd.DataFrame | str,
    target_column: str,
    hyperparameters: dict,
    test_size: float,
    metadata_columns: list = None,
    save_interval_models: bool = False,
    save_path: str = None,
    save_final_model: bool = False,
    plot_feat_importance: bool = False,
    batch_size: int = 2,
    n_jobs: int = 1,
    cv_seeds: list = None,
    search_seeds: list = None,
    split_seeds: list = None,
    final_model_seed: int = None,
    final_model_name: str = "final_model",
    save_training_ids_only: bool = True,
):
    """
    Train and evaluate a model using repeated train/test splits.

    Performs hyperparameter optimization for each resample,
    aggregates performance metrics, and fits a final model
    on the complete training dataset.

    Model-specific behavior is provided by subclass methods.

    Returns
    -------
    final_model : sklearn estimator
        Final fitted estimator or pipeline.
    best_params : dict
        Aggregated best hyperparameters across resamples.
    performance_dict : dict
        Aggregated performance metrics.
    feat_importance_df : pd.DataFrame or None
        Average feature importance across resamples,
        if supported by the estimator.
    """

    if n_resamples < 1 or batch_size < 1:
        raise ValueError("n_resamples and batch_size must be positive")

    if (
        save_interval_models or save_final_model or plot_feat_importance
    ) and save_path is None:
        raise ValueError("save_path is required when saving outputs")

    # Prepare random seeds
    cv_seeds, search_seeds, split_seeds = self._prepare_seed_lists(
        n_resamples=n_resamples,
        cv_seeds=cv_seeds,
        search_seeds=search_seeds,
        split_seeds=split_seeds,
    )

    final_model_seed = self._resolve_final_seed(final_model_seed)

    # Prepare features and targets
    features, targets = self._prepare_data(
        data=data,
        target_column=target_column,
        metadata_columns=metadata_columns,
    )

    # Prepare output directories
    if save_path is not None:
        save_path = Path(save_path)

    if save_interval_models:
        self.interval_path = save_path / "all_resample_data"
        self.interval_path.mkdir(parents=True, exist_ok=True)
    else:
        self.interval_path = None

    # Process batches of resamples
    def _process_batch(batch_indices):
        results_batch = []

        for n in batch_indices:
            result = self._fit_model_and_evaluate(
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
            )

            results_batch.append(result)

        return results_batch

    n_batches = (n_resamples + batch_size - 1) // batch_size

    batches = [
        range(
            i * batch_size,
            min((i + 1) * batch_size, n_resamples),
        )
        for i in range(n_batches)
    ]

    results_batches = Parallel(n_jobs=n_jobs)(
        delayed(_process_batch)(batch) for batch in batches
    )

    results = [result for batch in results_batches for result in batch]

    # Unpack the standardized resample results
    (
        best_params_ls,
        self.performance_ls,
        feat_importance_ls,
        self.resample_number_ls,
        self.true_vals_ls,
        self.pred_vals_ls,
    ) = zip(*results)

    # Derive final hyperparameters
    self.best_params_df = pd.DataFrame(best_params_ls)

    self.best_params_df = pd.DataFrame(best_params_ls)

    # Aggregate model-specific performance
    self.performance_dict = self._aggregate_performance(self.performance_ls)

    # Aggregate feature importance, when available
    feat_importance_df = None

    if all(df is not None for df in feat_importance_ls):
        importance_df = pd.concat(
            [df.set_index("Feature")["Importance"] for df in feat_importance_ls],
            axis=1,
        )

        feat_importance_df = (
            importance_df.mean(axis=1)
            .rename("Importance")
            .reset_index()
            .sort_values("Importance", ascending=False)
        )

        if plot_feat_importance:
            self._plot_feature_importance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

    # Perform final hyperparameter search using all training data
    final_cv, _ = self._set_inner_cv(
        cv_kwargs=self.cv_kwargs,
        cv_seed=final_model_seed,
    )

    final_estimator = self._create_estimator()

    final_estimator = self._set_final_model_seed(
        final_estimator,
        final_model_seed,
    )

    final_search, _ = self._set_hyperparameter_search(
        search_kwargs=self.hp_search_kwargs,
        hyperparameters=hyperparameters,
        estimator=final_estimator,
        search_seed=final_model_seed,
    )

    final_search.fit(features, targets.values.ravel())

    # Extract the optimized parameters and fitted estimator
    best_params = self._cast_best_params(final_search.best_params_)

    self.final_model = final_search.best_estimator_

    self.logger.info("Final model trained.\n")

    # Optionally save the final model and training information
    if save_final_model:
        training_path = save_path / "training_data"
        training_path.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            self.final_model,
            training_path / f"{final_model_name}.pkl.gz",
            compress=("gzip", 3),
        )

        with open(training_path / "performance_stats.json", "w") as file:
            json.dump(self.performance_dict, file, indent=4)

        with open(training_path / "best_params.json", "w") as file:
            json.dump(best_params, file, indent=4)

        if save_training_ids_only:
            targets.index.to_frame(
                index=False,
                name="ID",
            ).to_csv(
                training_path / "training_ids.csv",
                index=False,
            )
        else:
            features.to_csv(
                training_path / "training_features.csv.gz",
                index_label="ID",
                compression="gzip",
            )

            targets.to_csv(
                training_path / "training_targets.csv.gz",
                index_label="ID",
                compression="gzip",
            )

    self.logger.info(f"Model performance:\n{self.performance_dict}\n")

    return (
        self.final_model,
        best_params,
        self.performance_dict,
        feat_importance_df,
    )


def predict(
    self,
    feature_data: pd.DataFrame | str,
    prediction_col: str,
    final_model=None,
    save_preds: bool = False,
    save_path: str = None,
    filename: str = None,
):
    """
    Generate predictions using a fitted model.

    Parameters
    ----------
    feature_data : pd.DataFrame or str
        Features to predict, or a CSV path with an ID column.
    prediction_col : str
        Name of the prediction column.
    final_model : estimator, str, Path, or None
        Fitted estimator, path to saved estimator, or None to use
        self.final_model.
    save_preds : bool, default=False
        Whether to save predictions.
    save_path : str or None, default=None
        Directory for saving predictions.
    filename : str or None, default=None
        Output filename without extension.

    Returns
    -------
    pd.DataFrame
        Predictions indexed by ID, with optional probability
        and uncertainty columns.
    """

    # Load prediction features
    if isinstance(feature_data, (str, Path)):
        feature_data = pd.read_csv(
            feature_data,
            index_col="ID",
        )

    # Resolve fitted estimator
    if isinstance(final_model, (str, Path)):
        model = joblib.load(final_model)
    elif final_model is not None:
        model = final_model
    else:
        model = self.final_model

    # Prepare features
    feature_data = feature_data.apply(
        pd.to_numeric,
        errors="coerce",
    )

    if hasattr(model, "feature_names_in_"):
        feature_data = feature_data.reindex(
            columns=model.feature_names_in_,
        )

    if feature_data.isna().any().any():
        raise ValueError(
            "Prediction data contains missing or non-numeric "
            "values in required feature columns."
        )

    # Generate predictions
    preds_df = pd.DataFrame(index=feature_data.index)

    preds_df[prediction_col] = model.predict(feature_data)

    # Add optional classification probabilities
    probabilities = self._get_probabilities(
        model,
        feature_data,
    )

    if probabilities is not None:
        preds_df = preds_df.join(probabilities)

    # Add optional model-specific uncertainty
    uncertainty = self._get_uncertainty(
        model,
        feature_data,
    )

    if uncertainty is not None:
        preds_df["Uncertainty"] = uncertainty

    # Save predictions when requested
    if save_preds:
        if save_path is None or filename is None:
            raise ValueError("Both save_path and filename must be provided.")

        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        preds_df.to_csv(
            save_path / f"{filename}.csv.gz",
            compression="gzip",
            index_label="ID",
        )

    return preds_df


class RegressorBase(ModelBase):
    def _calculate_performance(
        self, feature_test: pd.DataFrame, target_test: pd.DataFrame, best_model
    ):
        true = np.asarray(target_test, dtype=float).ravel()
        pred = np.asarray(best_model.predict(feature_test)).ravel()

        errors = true - pred
        bias = np.mean(errors)
        sdep = (np.mean((true - pred - (np.mean(true - pred))) ** 2)) ** 0.5
        mse = mean_squared_error(true, pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(true, pred)

        try:
            r_pearson, p_pearson = pearsonr(true, pred)
        except Exception as e:
            self.logger.warning(f"Pearson correlation failed: {e}\n")
            r_pearson, p_pearson = None, None

        try:
            r_spearman, _ = spearmanr(true, pred)
        except Exception as e:
            self.logger.warning(f"Spearman correlation failed: {e}\n")
            r_spearman = None

        performance = {
            "bias": bias,
            "sdep": sdep,
            "mse": mse,
            "rmse": rmse,
            "r2": r2,
            "r_pearson": r_pearson,
            "p_pearson": p_pearson,
            "r_spearman": r_spearman,
        }

        pred_df = pd.DataFrame({"Prediction": pred}, index=feature_test.index)
        return performance, true, pred, pred_df


class ClassifierBase(ModelBase):
    def _calculate_performance(
        self,
        feature_test: pd.DataFrame,
        target_test: pd.DataFrame,
        best_model,
    ):
        true = np.asarray(target_test).ravel()
        pred = np.asarray(best_model.predict(feature_test)).ravel()

        pred_prob = None
        proba = None
        if hasattr(best_model, "predict_proba"):
            proba = np.asarray(best_model.predict_proba(feature_test))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                pred_prob = proba[:, 1]

        accuracy = accuracy_score(true, pred)
        balanced_accuracy = balanced_accuracy_score(true, pred)
        mcc = matthews_corrcoef(true, pred)

        sensitivity = np.nan
        specificity = np.nan
        ppv = np.nan
        npv = np.nan
        auc = np.nan

        cm = confusion_matrix(true, pred)
        if cm.shape == (2, 2):
            true_neg, false_pos, false_neg, true_pos = cm.ravel()
            sensitivity = (
                true_pos / (true_pos + false_neg) if (true_pos + false_neg) else np.nan
            )
            specificity = (
                true_neg / (true_neg + false_pos) if (true_neg + false_pos) else np.nan
            )
            ppv = (
                true_pos / (true_pos + false_pos) if (true_pos + false_pos) else np.nan
            )
            npv = (
                true_neg / (true_neg + false_neg) if (true_neg + false_neg) else np.nan
            )

        if proba is not None:
            try:
                if proba.ndim == 2 and proba.shape[1] == 2:
                    auc = roc_auc_score(true, proba[:, 1])
                elif self.use_multiclass:
                    auc = roc_auc_score(true, proba, multi_class="ovr")
            except Exception:
                auc = np.nan

        performance = {
            "acc": accuracy,
            "bal_acc": balanced_accuracy,
            "sens": sensitivity,
            "spec": specificity,
            "ppv": ppv,
            "npv": npv,
            "auc": auc,
            "mcc": mcc,
        }

        pred_df = pd.DataFrame({"Prediction": pred}, index=feature_test.index)
        if pred_prob is not None:
            pred_df["Pred_Prob"] = pred_prob
        return performance, true, pred, pred_df, pred_prob
