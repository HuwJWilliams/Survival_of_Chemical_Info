"""Random Forest model adapters for the shared training/evaluation framework.

Requires model_base_generalized.py in the Python import path. If that file has
been renamed to model_base.py, update the import below accordingly.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from model_base import ClassifierBase, RegressorBase


class RFRegressor(RegressorBase):
    """Random Forest regressor with feature importance and tree-spread uncertainty."""

    def _create_estimator(self):
        return RandomForestRegressor(random_state=self.random_seed)

    def _get_feature_importance(self, model, feature_names):
        return pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

    def _get_uncertainty(self, model, features):
        predictions = np.stack(
            [tree.predict(features.to_numpy()) for tree in model.estimators_]
        )
        return np.std(predictions, axis=0)


class RFClassifier(ClassifierBase):
    """Binary or multiclass Random Forest classifier with tree-spread uncertainty."""

    def _create_estimator(self):
        return RandomForestClassifier(random_state=self.random_seed)

    def _get_feature_importance(self, model, feature_names):
        return pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": model.feature_importances_,
            }
        ).sort_values("Importance", ascending=False)

    def _get_uncertainty(self, model, features):
        tree_probas = np.stack(
            [tree.predict_proba(features.to_numpy()) for tree in model.estimators_]
        )
        if len(model.classes_) == 2:
            # Across-tree variation in P(classes_[1]).
            probabilities = tree_probas[:, :, 1]
        else:
            # Match the previous RFMultiClassifier convention: max per-tree P(class).
            probabilities = np.max(tree_probas, axis=2)
        return np.std(probabilities, axis=0)


class RFMultiClassifier(RFClassifier):
    """Optional compatibility class; ClassifierBase already detects multiclass."""

    pass
