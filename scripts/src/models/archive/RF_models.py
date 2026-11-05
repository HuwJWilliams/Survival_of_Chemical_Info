import inspect
import json
import logging as log
from pathlib import Path
from typing import Callable, Union

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
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

from model_base import ModelBase


class RFRegressor(ModelBase):
    def __init__(
        self,
        cv_function: Callable,
        hp_search_function: Callable,
        cv_kwargs: dict,
        hp_search_kwargs: dict,
        log_level: int = log.INFO,
        random_seed: int = None,
    ):
        super().__init__(
            cv_function=cv_function,
            hp_search_function=hp_search_function,
            cv_kwargs=cv_kwargs,
            hp_search_kwargs=hp_search_kwargs,
            log_level=log_level,
            random_seed=random_seed,
        )
        self.logger.info("RFRegressor initialised.\n")

    def _calculate_performance(
        self,
        feature_test: pd.DataFrame,
        target_test: pd.DataFrame,
        best_rf,
    ):
        true = np.asarray(target_test, dtype=float).ravel()
        pred = np.asarray(best_rf.predict(feature_test)).ravel()

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

        resample_n = resample_n + 1

        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features,
            targets,
            test_size=test_size,
            random_state=split_seed,
        )
        tar_tr = tar_tr.values.ravel() if isinstance(tar_tr, pd.DataFrame) else tar_tr
        tar_te = tar_te.values.ravel() if isinstance(tar_te, pd.DataFrame) else tar_te

        model_estimator = RandomForestRegressor(random_state=self.random_seed)
        self.inner_cv, _ = self._set_inner_cv(cv_kwargs=self.cv_kwargs, cv_seed=cv_seed)
        model, _ = self._set_hyperparameter_search(
            search_kwargs=self.hp_search_kwargs,
            hyperparameters=hyperparameters,
            estimator=model_estimator,
            search_seed=search_seed,
        )

        model.fit(feat_tr, tar_tr)
        best_rf = model.best_estimator_

        performance, true, pred, pred_df = self._calculate_performance(
            target_test=tar_te,
            feature_test=feat_te,
            best_rf=best_rf,
        )

        if save_interval_models:
            joblib.dump(best_rf, f"{save_path}/model_resample_{resample_n}.pkl")
            pred_df.to_csv(
                f"{save_path}/preds_resample_{resample_n}.csv.gz",
                compression="gzip",
                index_label="ID",
            )
            with open(
                f"{save_path}/performance_stats_resample_{resample_n}.json", "w"
            ) as file:
                json.dump(performance, file, indent=4)

            feat_importance_df = pd.DataFrame(
                {
                    "Feature": features.columns,
                    "Importance": best_rf.feature_importances_,
                }
            ).sort_values(by="Importance", ascending=False)
            feat_importance_df.to_csv(
                f"{save_path}/feature_importance_resample_{resample_n}.csv",
                index=False,
            )

        return (
            model.best_params_,
            performance,
            best_rf.feature_importances_,
            resample_n,
            true,
            pred,
        )

    def trainRFRegressor(
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
        final_rf_seed: int = None,
        final_model_name: str = "final_model",
        save_training_ids_only: bool = True,
    ):
        cv_seeds, search_seeds, split_seeds = self._prepare_seed_lists(
            n_resamples=n_resamples,
            cv_seeds=cv_seeds,
            search_seeds=search_seeds,
        )
        final_rf_seed = self._resolve_final_seed(final_rf_seed)

        features, targets = self._prepare_data(
            data=data,
            target_column=target_column,
            metadata_columns=metadata_columns,
        )

        if save_interval_models:
            self.interval_path = Path(f"{save_path}/all_resample_data/")
            self.interval_path.mkdir(parents=True, exist_ok=True)
        else:
            self.interval_path = save_path

        def _process_batch(batch_indices: list):
            results_batch = []
            for n in batch_indices:
                results_batch.append(
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
                    )
                )
            return results_batch

        n_batches = (n_resamples + batch_size - 1) // batch_size
        batches = [
            range(i * batch_size, min((i + 1) * batch_size, n_resamples))
            for i in range(n_batches)
        ]

        results_batches = Parallel(n_jobs=n_jobs)(
            delayed(_process_batch)(batch) for batch in batches
        )
        results = [result for batch in results_batches for result in batch]

        (
            best_params_ls,
            self.performance_ls,
            feat_importance_ls,
            self.resample_number_ls,
            self.true_vals_ls,
            self.pred_vals_ls,
        ) = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self._derive_best_params(best_params_ls)
        best_params["random_state"] = final_rf_seed
        best_params = self._cast_best_params(best_params)

        self.performance_dict = {
            "Bias": round(
                float(np.mean([perf["bias"] for perf in self.performance_ls])), 4
            ),
            "SDEP": round(
                float(np.mean([perf["sdep"] for perf in self.performance_ls])), 4
            ),
            "MSE": round(
                float(np.mean([perf["mse"] for perf in self.performance_ls])), 4
            ),
            "RMSE": round(
                float(np.mean([perf["rmse"] for perf in self.performance_ls])), 4
            ),
            "r2": round(
                float(np.mean([perf["r2"] for perf in self.performance_ls])), 4
            ),
            "Pearson_r": round(
                float(np.mean([perf["r_pearson"] for perf in self.performance_ls])), 4
            ),
            "Pearson_p": round(
                float(np.mean([perf["p_pearson"] for perf in self.performance_ls])), 4
            ),
        }

        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": features.columns.tolist(), "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)

        if plot_feat_importance:
            self._plot_feature_importance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

        self.final_rf = RandomForestRegressor(**best_params)
        self.final_rf.fit(features, targets.values.ravel())
        self.logger.info("Final RandomForestRegressor model trained.\n")

        if save_final_model:
            Path(f"{save_path}/training_data/").mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"Saving final model to:\n{save_path}/{final_model_name}.pkl.gz\n"
            )
            joblib.dump(
                self.final_rf,
                f"{save_path}/training_data/{final_model_name}.pkl.gz",
                compress=("gzip", 3),
            )

            with open(f"{save_path}/training_data/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file, indent=4)

            with open(f"{save_path}/training_data/best_params.json", "w") as file:
                json.dump(best_params, file, indent=4)

            if not save_training_ids_only:
                features.to_csv(
                    f"{save_path}/training_data/training_features.csv.gz",
                    index_label="ID",
                    compression="gzip",
                )
                targets.to_csv(
                    f"{save_path}/training_data/training_targets.csv.gz",
                    index_label="ID",
                    compression="gzip",
                )
            else:
                targets.index.to_frame(index=False, name="ID").to_csv(
                    f"{save_path}/training_data/training_ids.csv",
                    index=False,
                )

        self.logger.info(
            f"Performance of final RandomForestRegressor model:\n{self.performance_dict}\n"
        )

        return self.final_rf, best_params, self.performance_dict, feat_importance_df

    def predictRFRegressor(
        self,
        feature_data: Union[pd.DataFrame, str],
        prediction_col: str,
        final_rf: Union[RandomForestRegressor, str, Path],
        save_preds: bool = False,
        save_path: str = None,
        filename: str = None,
    ):
        if isinstance(feature_data, str):
            feature_data = pd.read_csv(feature_data, index_col="ID")

        if isinstance(final_rf, (str, Path)):
            rf_model = joblib.load(final_rf)
        elif isinstance(final_rf, RandomForestRegressor):
            rf_model = final_rf
        else:
            rf_model = self.final_rf

        feature_data = feature_data.apply(pd.to_numeric, errors="coerce")
        if hasattr(rf_model, "feature_names_in_"):
            feature_data = feature_data.reindex(columns=rf_model.feature_names_in_)
        else:
            feature_data = feature_data.dropna(axis=1)

        preds_df = pd.DataFrame(index=feature_data.index)
        preds_df[prediction_col] = rf_model.predict(feature_data)

        all_tree_preds = np.stack(
            [tree.predict(feature_data.to_numpy()) for tree in rf_model.estimators_]
        )
        preds_df["Uncertainty"] = np.std(all_tree_preds, axis=0)

        if save_preds:
            if save_path is None or filename is None:
                raise ValueError(
                    "Both save_path and filename must be provided to save predictions"
                )
            preds_df.to_csv(
                f"{save_path}/{filename}.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        return preds_df


class RFClassifier(ModelBase):
    def __init__(
        self,
        cv_function: Callable,
        hp_search_function: Callable,
        cv_kwargs: dict,
        hp_search_kwargs: dict,
        log_level: int = log.INFO,
        random_seed: int = None,
        use_multiclass: bool = False,
    ):
        super().__init__(
            cv_function=cv_function,
            hp_search_function=hp_search_function,
            cv_kwargs=cv_kwargs,
            hp_search_kwargs=hp_search_kwargs,
            log_level=log_level,
            random_seed=random_seed,
        )
        self.use_multiclass = use_multiclass
        self.logger.info("RFClassifier initialised.\n")

    def _calculate_performance(
        self,
        feature_test: pd.DataFrame,
        target_test: pd.DataFrame,
        best_rf,
    ):
        true = np.asarray(target_test).ravel()
        pred = np.asarray(best_rf.predict(feature_test)).ravel()

        pred_prob = None
        proba = None
        if hasattr(best_rf, "predict_proba"):
            proba = np.asarray(best_rf.predict_proba(feature_test))
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
        resample_n = resample_n + 1
        rf_seed = split_seed

        feat_tr, feat_te, tar_tr, tar_te = train_test_split(
            features,
            targets,
            test_size=test_size,
            random_state=rf_seed,
        )

        tar_tr = tar_tr.values.ravel() if isinstance(tar_tr, pd.DataFrame) else tar_tr
        tar_te = tar_te.values.ravel() if isinstance(tar_te, pd.DataFrame) else tar_te

        model_estimator = RandomForestClassifier(random_state=self.random_seed)
        self.inner_cv, _ = self._set_inner_cv(cv_kwargs=self.cv_kwargs, cv_seed=cv_seed)
        model, _ = self._set_hyperparameter_search(
            search_kwargs=self.hp_search_kwargs,
            hyperparameters=hyperparameters,
            estimator=model_estimator,
            search_seed=search_seed,
        )

        model.fit(feat_tr, tar_tr)
        best_rf = model.best_estimator_

        performance, true, pred, pred_df, pred_prob = self._calculate_performance(
            target_test=tar_te,
            feature_test=feat_te,
            best_rf=best_rf,
        )

        if save_interval_models:
            joblib.dump(best_rf, f"{save_path}/model_resample_{resample_n}.pkl")
            pred_df.to_csv(
                f"{save_path}/preds_resample_{resample_n}.csv.gz",
                compression="gzip",
                index_label="ID",
            )
            with open(
                f"{save_path}/performance_stats_resample_{resample_n}.json", "w"
            ) as file:
                json.dump(performance, file, indent=4)

            feat_importance_df = pd.DataFrame(
                {
                    "Feature": features.columns,
                    "Importance": best_rf.feature_importances_,
                }
            ).sort_values(by="Importance", ascending=False)
            feat_importance_df.to_csv(
                f"{save_path}/feature_importance_resample_{resample_n}.csv",
                index=False,
            )

        return (
            model.best_params_,
            performance,
            best_rf.feature_importances_,
            resample_n,
            true,
            pred,
            pred_prob,
        )

    def trainRFClassifier(
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
        final_rf_seed: int = None,
        final_model_name: str = "final_model",
    ):
        cv_seeds, search_seeds, split_seeds = self._prepare_seed_lists(
            n_resamples=n_resamples,
            cv_seeds=cv_seeds,
            search_seeds=search_seeds,
        )
        final_rf_seed = self._resolve_final_seed(final_rf_seed)

        features, targets = self._prepare_data(
            data=data,
            target_column=target_column,
            metadata_columns=metadata_columns,
        )

        if save_interval_models:
            self.interval_path = Path(f"{save_path}/all_resample_data/")
            self.interval_path.mkdir(parents=True, exist_ok=True)
        else:
            self.interval_path = save_path

        def _process_batch(batch_indices: list):
            results_batch = []
            for n in batch_indices:
                results_batch.append(
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
                    )
                )
            return results_batch

        n_batches = (n_resamples + batch_size - 1) // batch_size
        batches = [
            range(i * batch_size, min((i + 1) * batch_size, n_resamples))
            for i in range(n_batches)
        ]

        results_batches = Parallel(n_jobs=n_jobs)(
            delayed(_process_batch)(batch) for batch in batches
        )
        results = [result for batch in results_batches for result in batch]

        (
            best_params_ls,
            self.performance_ls,
            feat_importance_ls,
            self.resample_number_ls,
            self.true_vals_ls,
            self.pred_vals_ls,
            self.pred_prob_ls,
        ) = zip(*results)

        self.best_params_df = pd.DataFrame(best_params_ls)
        best_params = self._derive_best_params(best_params_ls)
        best_params["random_state"] = final_rf_seed
        best_params = self._cast_best_params(best_params)

        if not self.use_multiclass:
            self.performance_dict = {
                "Accuracy": round(
                    float(
                        np.nanmean(
                            [perf.get("acc", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "Balanced_Accuracy": round(
                    float(
                        np.nanmean(
                            [
                                perf.get("bal_acc", np.nan)
                                for perf in self.performance_ls
                            ]
                        )
                    ),
                    4,
                ),
                "Sensitivity": round(
                    float(
                        np.nanmean(
                            [perf.get("sens", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "Specificity": round(
                    float(
                        np.nanmean(
                            [perf.get("spec", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "PPV": round(
                    float(
                        np.nanmean(
                            [perf.get("ppv", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "NPV": round(
                    float(
                        np.nanmean(
                            [perf.get("npv", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "AUC": round(
                    float(
                        np.nanmean(
                            [perf.get("auc", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "MCC": round(
                    float(
                        np.nanmean(
                            [perf.get("mcc", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
            }
        else:
            self.performance_dict = {
                "Accuracy": round(
                    float(
                        np.nanmean(
                            [perf.get("acc", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "Balanced_Accuracy": round(
                    float(
                        np.nanmean(
                            [
                                perf.get("bal_acc", np.nan)
                                for perf in self.performance_ls
                            ]
                        )
                    ),
                    4,
                ),
                "F1_macro": round(
                    float(
                        np.nanmean(
                            [
                                perf.get("f1_macro", np.nan)
                                for perf in self.performance_ls
                            ]
                        )
                    ),
                    4,
                ),
                "AUC_OVR": round(
                    float(
                        np.nanmean(
                            [perf.get("auc", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
                "MCC": round(
                    float(
                        np.nanmean(
                            [perf.get("mcc", np.nan) for perf in self.performance_ls]
                        )
                    ),
                    4,
                ),
            }

        avg_feat_importance = np.mean(feat_importance_ls, axis=0)
        feat_importance_df = pd.DataFrame(
            {"Feature": features.columns.tolist(), "Importance": avg_feat_importance}
        ).sort_values(by="Importance", ascending=False)

        if plot_feat_importance:
            self._plot_feature_importance(
                feat_importance_df=feat_importance_df,
                save_data=True,
                save_path=save_path,
                filename="feature_importance_plot",
            )

        self.final_rf = RandomForestClassifier(**best_params)
        self.final_rf.fit(features, targets.values.ravel())
        self.logger.info("Final RandomForestClassifier model trained.\n")

        if save_final_model:
            Path(f"{save_path}/training_data/").mkdir(parents=True, exist_ok=True)

            self.logger.info(
                f"Saving final model to:\n{save_path}/{final_model_name}.pkl.gz\n"
            )
            joblib.dump(
                self.final_rf,
                f"{save_path}/training_data/{final_model_name}.pkl.gz",
                compress=("gzip", 3),
            )

            with open(f"{save_path}/training_data/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file, indent=4)

            with open(f"{save_path}/training_data/best_params.json", "w") as file:
                json.dump(best_params, file, indent=4)

            features.to_csv(
                f"{save_path}/training_data/training_features.csv.gz",
                index_label="ID",
                compression="gzip",
            )
            targets.to_csv(
                f"{save_path}/training_data/training_targets.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        self.logger.info(
            f"Performance of final RandomForestClassifier model:\n{self.performance_dict}\n"
        )

        return self.final_rf, best_params, self.performance_dict, feat_importance_df

    def predictRFClassifier(
        self,
        feature_data: Union[pd.DataFrame, str],
        prediction_col: str,
        final_rf: Union[RandomForestClassifier, str, Path],
        save_preds: bool = False,
        save_path: str = None,
        filename: str = None,
    ):
        if isinstance(feature_data, str):
            feature_data = pd.read_csv(feature_data, index_col="ID")

        if isinstance(final_rf, (str, Path)):
            rf_model = joblib.load(final_rf)
        elif isinstance(final_rf, RandomForestClassifier):
            rf_model = final_rf
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame(index=feature_data.index)
        preds_df[prediction_col] = rf_model.predict(feature_data)

        if hasattr(rf_model, "predict_proba"):
            proba = np.asarray(rf_model.predict_proba(feature_data))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                preds_df["Pred_Prob"] = proba[:, 1]
            else:
                preds_df["Pred_Prob"] = np.max(proba, axis=1)

            all_tree_probs = []
            for tree in rf_model.estimators_:
                tree_proba = np.asarray(tree.predict_proba(feature_data.to_numpy()))
                if tree_proba.ndim == 2 and tree_proba.shape[1] >= 2:
                    all_tree_probs.append(tree_proba[:, 1])
                else:
                    all_tree_probs.append(np.max(tree_proba, axis=1))
            preds_df["Uncertainty"] = np.std(np.stack(all_tree_probs), axis=0)

        if save_preds:
            if save_path is None or filename is None:
                raise ValueError(
                    "Both save_path and filename must be provided to save predictions"
                )

            preds_df.to_csv(
                f"{save_path}/{filename}.csv.gz",
                index_label="ID",
                compression="gzip",
            )

        return preds_df


class RFMultiClassifier(RFClassifier):
    def __init__(
        self,
        cv_function: Callable,
        hp_search_function: Callable,
        cv_kwargs: dict,
        hp_search_kwargs: dict,
        log_level: int = log.INFO,
        random_seed: int = None,
    ):
        super().__init__(
            cv_function=cv_function,
            hp_search_function=hp_search_function,
            cv_kwargs=cv_kwargs,
            hp_search_kwargs=hp_search_kwargs,
            log_level=log_level,
            random_seed=random_seed,
            use_multiclass=True,
        )
        self.logger.info("RFMultiClassifier initialised.\n")

    def _calculate_performance(
        self,
        feature_test: pd.DataFrame,
        target_test: pd.DataFrame,
        best_rf,
    ):
        true = np.asarray(target_test).ravel()
        pred = np.asarray(best_rf.predict(feature_test)).ravel()

        proba = np.asarray(best_rf.predict_proba(feature_test))

        accuracy = accuracy_score(true, pred)
        bal_acc = balanced_accuracy_score(true, pred)
        mcc = matthews_corrcoef(true, pred)
        f1_macro = f1_score(true, pred, average="macro")

        auc_ovr = np.nan
        try:
            auc_ovr = roc_auc_score(true, proba, multi_class="ovr")
        except Exception:
            auc_ovr = np.nan

        performance = {
            "acc": accuracy,
            "auc": auc_ovr,
            "mcc": mcc,
            "bal_acc": bal_acc,
            "f1_macro": f1_macro,
        }

        pred_df = pd.DataFrame({"Prediction": pred}, index=feature_test.index)
        return performance, true, pred, pred_df, None

    def trainRFMultiClassifier(
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
        final_rf_seed: int = None,
        final_model_name: str = "final_model",
    ):
        final_model, best_params, _, feat_importance_df = super().trainRFClassifier(
            n_resamples=n_resamples,
            data=data,
            target_column=target_column,
            hyperparameters=hyperparameters,
            test_size=test_size,
            metadata_columns=metadata_columns,
            save_interval_models=save_interval_models,
            save_path=save_path,
            save_final_model=save_final_model,
            plot_feat_importance=plot_feat_importance,
            batch_size=batch_size,
            n_jobs=n_jobs,
            cv_seeds=cv_seeds,
            search_seeds=search_seeds,
            final_rf_seed=final_rf_seed,
            final_model_name=final_model_name,
        )

        self.performance_dict = {
            "Accuracy": round(
                float(np.nanmean([perf["acc"] for perf in self.performance_ls])), 4
            ),
            "Balanced_Accuracy": round(
                float(
                    np.nanmean(
                        [perf.get("bal_acc", np.nan) for perf in self.performance_ls]
                    )
                ),
                4,
            ),
            "F1_macro": round(
                float(
                    np.nanmean(
                        [perf.get("f1_macro", np.nan) for perf in self.performance_ls]
                    )
                ),
                4,
            ),
            "AUC_OVR": round(
                float(np.nanmean([perf["auc"] for perf in self.performance_ls])), 4
            ),
            "MCC": round(
                float(np.nanmean([perf["mcc"] for perf in self.performance_ls])), 4
            ),
        }

        if save_final_model:
            with open(f"{save_path}/training_data/performance_stats.json", "w") as file:
                json.dump(self.performance_dict, file, indent=4)

        self.logger.info(
            f"Performance of final RandomForestMultiClassifier model:\n{self.performance_dict}\n"
        )
        return final_model, best_params, self.performance_dict, feat_importance_df

    def predictRFMultiClassifier(
        self,
        feature_data: Union[pd.DataFrame, str],
        prediction_col: str,
        final_rf: Union[RandomForestClassifier, str, Path],
        save_preds: bool = False,
        save_path: str = None,
        filename: str = None,
    ):
        if isinstance(feature_data, str):
            feature_data = pd.read_csv(feature_data, index_col="ID")

        if isinstance(final_rf, (str, Path)):
            rf_model = joblib.load(final_rf)
        elif isinstance(final_rf, RandomForestClassifier):
            rf_model = final_rf
        else:
            rf_model = self.final_rf

        preds_df = pd.DataFrame(index=feature_data.index)
        preds_df[prediction_col] = rf_model.predict(feature_data)

        proba = np.asarray(rf_model.predict_proba(feature_data))
        classes = list(rf_model.classes_)
        for i, cls in enumerate(classes):
            preds_df[f"Pred_Prob_{cls}"] = proba[:, i]
        preds_df["Pred_Prob_Max"] = np.max(proba, axis=1)

        all_tree_max_probs = []
        for tree in rf_model.estimators_:
            tree_proba = np.asarray(tree.predict_proba(feature_data.to_numpy()))
            all_tree_max_probs.append(np.max(tree_proba, axis=1))
        preds_df["Uncertainty"] = np.std(np.stack(all_tree_max_probs), axis=0)

        if save_preds:
            if save_path is None or filename is None:
                raise ValueError(
                    "Both save_path and filename must be provided to save predictions"
                )
            preds_df.to_csv(
                f"{save_path}/{filename}.csv.gz",
                index_label="ID",
                compression="gzip",
            )
        return preds_df
