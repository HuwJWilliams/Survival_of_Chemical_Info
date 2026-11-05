from sklearn.linear_model import Ridge
from sklearn.svm import SVR

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
       return Pipeline([
           ('scaler', StandardScaler()),
           ("svr", SVR())
       ])