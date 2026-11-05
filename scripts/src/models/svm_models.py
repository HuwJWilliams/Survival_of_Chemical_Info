from sklearn.svm import SVR
from model_base import RegressorBase
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class SVRegressor(RegressorBase):

    def _create_estimator(self):
        return Pipeline([("scaler", StandardScaler()), ("svr", SVR())])
