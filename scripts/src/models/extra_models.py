from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor as SklearnMLPRegressor

from model_base import RegressorBase


class RidgeRegressor(RegressorBase):
    """Ridge regression with standardized features."""

    def _create_estimator(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("ridge", Ridge(random_state=self.random_seed)),
            ]
        )


class SVRegressor(RegressorBase):

    def _create_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("svr", SVR())])


class MLPRegressor(RegressorBase):
    """Multilayer perceptron regression with standardized features."""

    def _create_estimator(self):
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "mlp",
                    SklearnMLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        solver="adam",
                        max_iter=1000,
                        random_state=self.random_seed,
                        early_stopping=True,
                        n_iter_no_change=20,
                        tol=1e-4,
                        validation_fraction=0.1,
                    ),
                ),
            ]
        )
