from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_validate, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from .parent import Parent
import numpy as np

ALGORITHM_NAMES = [
    "Linear Regression",
    "SVR",
    "K-Neighbors Regressor",
    "Random Forest Regressor",
    "Decision Tree Regressor",
    "XGBoost Regressor",
    "GradientBoosting Regressor",
]
ALGORITHM_OBJECT = [
    LinearRegression(),
    SVR(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    XGBRegressor(),
    GradientBoostingRegressor(),
]

ALGORITHM_REG = dict(zip(ALGORITHM_NAMES, ALGORITHM_OBJECT))
METRICS_NAMES = [
    "mean_absolute_error",
    "mean_squared_error",
    "mean_absolute_percentage_error",
]

METRICS_OBJECT = [
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
]


class Regression(Parent):
    ALGORITHM = ALGORITHM_REG
    METRICS = dict(zip(METRICS_NAMES, METRICS_OBJECT))
    model_type = "Regression"

    def __str__(self) -> str:
        return "<Regression Object>"

    def fit(
        self, data: pd.DataFrame, target: str, features: list, norm=True, y_norm=False
    ):
        obj = super().fit(data, target, features, norm)
        if y_norm:
            obj.y = obj.y / max(obj.y)
        return obj

    def score(self, y_true, y_pred, metric="mean_absolute_percentage_error"):
        """
        Calculates the specified metric score between true and predicted target values.

        Parameters:
            y_true (array-like): The true target values.
            y_pred (array-like): The predicted target values.
            metric (str, optional): The metric to use for scoring. Default is 'mean_absolute_percentage_error'.
                                    Available metrics are defined in the METRICS dictionary.

        Returns:
            float: The calculated score based on the specified metric.

        Note:
            This method calculates the specified metric score between the true and predicted target values.
            Available metrics are defined in the METRICS dictionary.
        """
        err = self.METRICS[metric](y_true=y_true, y_pred=y_pred)
        return err

    def cross_val(metrics, X, y, estimator, cv):
        c = cross_validate(
            estimator, X, y, scoring="neg_mean_absolute_percentage_error", cv=cv
        )
        c["fit_time"] = c["fit_time"].mean()
        c["score_time"] = c["score_time"].mean()
        c["test_score"] = c["test_score"].mean()
        c["mean_absolute_percentage_error"] = abs(c.pop("test_score"))
        c["mean_squared_error"] = abs(
            cross_val_score(estimator, X, y, scoring="neg_mean_squared_error").mean()
        )
        c["mean_absolute_error"] = abs(
            cross_val_score(estimator, X, y, scoring="neg_mean_absolute_error").mean()
        )
        c["root_mean_squared_error"] = abs(
            cross_val_score(
                estimator, X, y, scoring="neg_root_mean_squared_error"
            ).mean()
        )
        return c

    def score_report(self, y_true, y_pred):
        """
        Generates a report containing various metric scores between true and predicted target values.

        Parameters:
            y_true (array-like): The true target values.
            y_pred (array-like): The predicted target values.

        Returns:
            dict: A dictionary containing metric scores calculated for each metric defined in the METRICS dictionary.

        Note:
            This method generates a report containing various metric scores between true and predicted target values.
            It calculates metric scores for each metric defined in the METRICS dictionary.
        """
        res = {}
        for i in self.METRICS:
            res[i] = self.score(y_true, y_pred, metric=i)
        return res

    def compare_model(
        self,
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        output="dict",
        train_val=False,
    ):
        """
        Compares the performance of different regression models using either train-test split or cross-validation.

        Parameters:
            X_train (array-like, optional): The feature matrix for training. Default is None.
            X_test (array-like, optional): The feature matrix for testing. Default is None.
            y_train (array-like, optional): The target vector for training. Default is None.
            y_test (array-like, optional): The target vector for testing. Default is None.
            output (str, optional): Specifies the format of the output. Default is 'dict'.
                                    Options: 'dict' (dictionary), 'dataframe' (DataFrame), 'only_mape'.
            train_val (bool, optional): Whether to compute performance metrics on both training and validation sets.
                                        Applicable only if X_train, X_test, y_train, and y_test are provided.
                                        Default is False.

        Returns:
            dict or DataFrame: A dictionary or DataFrame containing the performance metrics of the models.

        Note:
            This method compares the performance of different regression models using either train-test split or cross-validation.
            If train_val is True and X_train, X_test, y_train, and y_test are provided, it computes performance metrics
            on both training and validation sets. Otherwise, it computes performance metrics using cross-validation.
        """
        result = {}
        self.cross_validation = True
        if np.any(X_train) and np.any(X_test) and np.any(y_train) and np.any(y_test):
            self.cross_validation = False
            if train_val:
                for al in self.ALGORITHM:
                    report = {}
                    alg = self.ALGORITHM[al].fit(X_train, y_train)
                    print(f"{al} is run ...")
                    pred_train = alg.predict(X_train)
                    pred_val = alg.predict(X_test)
                    report["mae_train"] = self.score(
                        y_train, pred_train, metric="mean_absolute_error"
                    )
                    report["mae_val"] = self.score(
                        y_test, pred_val, "mean_absolute_error"
                    )
                    report["mse_train"] = self.score(
                        y_train, pred_train, "mean_squared_error"
                    )
                    report["mse_val"] = self.score(
                        y_test, pred_val, "mean_squared_error"
                    )
                    mape_train = self.score(
                        y_train, pred_train, "mean_absolute_percentage_error"
                    )
                    report["mape_train"] = mape_train
                    mape_val = self.score(
                        y_test, pred_val, "mean_absolute_percentage_error"
                    )
                    report["mape_val"] = mape_val
                    report["difference_mape"] = (mape_train - mape_val) * 100
                    result[al] = report
            else:
                for al in self.ALGORITHM:
                    alg = self.ALGORITHM[al].fit(X_train, y_train)
                    print(f"{al} is run ...")
                    y_pred = alg.predict(X_test)
                    report = self.score_report(y_test, y_pred)
                    result[al] = report
        else:
            kfold = KFold(n_splits=5, shuffle=True, random_state=self.seed)
            for al in self.ALGORITHM:
                alg = self.ALGORITHM[al]
                print(f"{al} is run ...")
                report = Regression.cross_val(
                    metrics=self.METRICS, estimator=alg, X=self.X, y=self.y, cv=kfold
                )
                result[al] = report
        self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "only_score":
            rest = {}
            for i in result:
                rest[i] = round(result[i]["mean_absolute_percentage_error"], 2)
            return rest
        else:
            return result
