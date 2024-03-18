from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.model_selection import cross_validate,cross_val_score,KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
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
    "GradientBoosting Regressor"
]
ALGORITHM_OBJECT = [
    LinearRegression(),
    SVR(),
    KNeighborsRegressor(),
    RandomForestRegressor(),
    DecisionTreeRegressor(),
    XGBRegressor(),
    GradientBoostingRegressor()
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

    def fit(self, data: pd.DataFrame, target: str, features: list, norm=True,y_norm=False):
        obj = super().fit(data, target, features, norm)
        if y_norm:
            obj.y = obj.y/max(obj.y)
        return obj

    def score(self,y_true,y_pred,metric='mean_absolute_percentage_error'):
        err = self.METRICS[metric](y_true=y_true,y_pred=y_pred)
        return err


    def cross_val(metrics,X,y,estimator,cv):
        c = cross_validate(estimator,X,y,scoring='neg_mean_absolute_percentage_error',cv=cv)
        c['fit_time'] = c['fit_time'].mean()
        c['score_time'] = c['score_time'].mean()
        c['test_score'] =  c['test_score'].mean()
        c['mape'] = c.pop('test_score')
        c['mse'] = cross_val_score(estimator,X,y,scoring='neg_mean_squared_error').mean()
        c['mae'] = cross_val_score(estimator,X,y,scoring='neg_mean_absolute_error').mean()
        c['rmse'] = cross_val_score(estimator,X,y,scoring='neg_root_mean_squared_error').mean()
        return c


    def score_report(self,y_true,y_pred):
        res = {}
        for i in self.METRICS:
            res[i] = self.score(y_true,y_pred,metric=i)
        return res

    def compare_model(self,X_train=None,X_test=None,y_train=None,y_test=None,output="dict",train_val=False):
        result = {}
        """
        default using cross_validation
        """
        self.cross_validation = True
        if np.any(X_train) and np.any(X_test) and np.any(y_train) and np.any(y_test):
            self.cross_validation = False
            if train_val:
                for al in self.ALGORITHM:
                    report = {}
                    alg = self.ALGORITHM[al].fit(X_train,y_train)
                    print(f'{al} is run ...')
                    pred_train = alg.predict(X_train)
                    pred_val = alg.predict(X_test)
                    report['mae_train'] = self.score(y_train,pred_train,metric='mean_absolute_error')
                    report['mae_val'] = self.score(y_test,pred_val,'mean_absolute_error')
                    report['mse_train'] = self.score(y_train,pred_train,'mean_squared_error')
                    report['mse_val'] = self.score(y_test,pred_val,'mean_squared_error')
                    mape_train = self.score(y_train,pred_train,'mean_absolute_percentage_error')
                    report['mape_train'] = mape_train
                    mape_val = self.score(y_test,pred_val,'mean_absolute_percentage_error')
                    report['mape_val'] = mape_val
                    report['difference_mape'] = (mape_train - mape_val) *100
                    result[al] = report
            else:
                for al in self.ALGORITHM:
                    alg = self.ALGORITHM[al].fit(X_train,y_train)
                    print(f'{al} is run ...')
                    y_pred = alg.predict(X_test)
                    report = self.score_report(y_test,y_pred)
                    result[al] = report
        else:
            kfold  = KFold(n_splits=5,shuffle=True,random_state=self.seed)
            for al in self.ALGORITHM:
                alg = self.ALGORITHM[al]
                print(f"{al} is run ...")
                report = Regression.cross_val(metrics=self.METRICS,estimator=alg,X=self.X,y=self.y,cv=kfold)
                result[al] = report
        self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "only_mape":
            rest = {}
            for i in result:
                rest[i] = round(result[i]["mean_absolute_percentage_error"],2)
            return rest
        else:
            return result

