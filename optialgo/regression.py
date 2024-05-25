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

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import Progress


class Regression(Parent):
    ALGORITHM = {
        "Linear Regression": LinearRegression(),
        "SVR": SVR(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Random Forest Regressor": RandomForestRegressor(),
        "Decision Tree Regressor": DecisionTreeRegressor(),
        "XGBoost Regressor": XGBRegressor(),
        "GradientBoosting Regressor": GradientBoostingRegressor(),
    }
    METRICS = {
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "mean_absolute_percentage_error": mean_absolute_percentage_error,
    }
    model_type = "Regression"

    def __str__(self) -> str:
        return "<Regression Object>"

    def score(self, y_true, y_pred, metric="mean_absolute_error"):
        """
        Calculate the evaluation metric for the given true and predicted values.

        This method computes a specified error metric based on the true and predicted values of the target variable.
        It uses the metrics stored in the `self.METRICS` dictionary.

        Parameters
        ----------
        y_true : array-like
            True values of the target variable.

        y_pred : array-like
            Predicted values of the target variable.

        metric : str, optional
            The metric to be used for evaluating the predictions.
            Default is "mean_absolute_error". Possible values include:
            - "mean_absolute_error"
            - "mean_squared_error"
            - "mean_absolute_percentage_error"
            - Other metrics defined in `self.METRICS`.

        Returns
        -------
        float
            The computed error metric value.

        Raises
        ------
        KeyError
            If the specified metric is not found in `self.METRICS`.

        Example
        -------
        >>> y_true = [3.0, -0.5, 2.0, 7.0]
        >>> y_pred = [2.5, 0.0, 2.0, 8.0]
        >>> regressor = Regressor(dataset, algorithm='linear_regression')
        >>> mae = regressor.score(y_true, y_pred, metric='mean_absolute_error')
        >>> print(mae)
        0.5
        """
        err = self.METRICS[metric](y_true=y_true, y_pred=y_pred)
        return err

    def __cross_val(metrics, X, y, estimator, cv):
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
        Generate a report of evaluation metrics for the given true and predicted values.

        This method computes various evaluation metrics defined in `self.METRICS` for the provided
        true and predicted values of the target variable. It returns a dictionary with the names
        of the metrics as keys and their computed values as values.

        Parameters
        ----------
        y_true : array-like
            True values of the target variable.

        y_pred : array-like
            Predicted values of the target variable.

        Returns
        -------
        dict
            A dictionary containing the names and values of the computed metrics. The keys are
            the metric names (as defined in `self.METRICS`) and the values are the corresponding
            metric values.

        Example
        -------
        >>> y_true = [3.0, -0.5, 2.0, 7.0]
        >>> y_pred = [2.5, 0.0, 2.0, 8.0]
        >>> regressor = Regressor(dataset, algorithm='linear_regression')
        >>> report = regressor.score_report(y_true, y_pred)
        >>> for metric, value in report.items():
        >>>     print(f"{metric}: {value:.4f}")
        mean_absolute_error: 0.5000
        mean_squared_error: 0.3750
        mean_absolute_percentage_error: 0.1271
        """
        res = {}
        for i in self.METRICS:
            res[i] = self.score(y_true, y_pred, metric=i)
        return res

    def compare_model(
        self,
        output="dict",
        train_val=False,
    ):
        """
        Compares multiple regression models based on various metrics and returns the results.

        This function evaluates a set of algorithms on the training and validation sets or using
        cross-validation, and compiles performance metrics. The results can be output in different
        formats, including dictionary, pandas DataFrame, or a formatted table.

        Parameters
        ----------
        output : str, optional
            The format of the output. It can be:
            - "dict": Returns the results as a dictionary (default).
            - "dataframe": Returns the results as a pandas DataFrame.
            - "table": Prints the results as a formatted table.
            - "only_score": Returns only the Mean Absolute Percentage Error (MAPE) scores in a simplified dictionary.

        train_val : bool, optional
            If True, the function evaluates the models using a train-validation split.
            If False, the function uses cross-validation. Default is False.

        Returns
        -------
        dict or pd.DataFrame or None
            The function returns results based on the `output` parameter:
            - If `output` is "dict", it returns a dictionary of the results.
            - If `output` is "dataframe", it returns a pandas DataFrame of the results.
            - If `output` is "table", it prints the results as a formatted table.
            - If `output` is "only_score", it returns a dictionary with only the MAPE scores.

        Raises
        ------
        ValueError
            If an invalid output type is provided.

        Example
        -------
        >>> dataset = Dataset(dataframe, norm=True, test_size=0.3, seed=123)
        >>> regressor = Regressor(dataset, algorithm='random_forest')
        >>> results = regressor.compare_model(output='dataframe', train_val=True)
        >>> print(results)
        """
        result = {}
        X_train, X_test, y_train, y_test = self.dataset.get_x_y()
        console = Console()

        with Progress() as progress:
            if train_val:
                title = "Train-Validation"
                task = progress.add_task(
                    "[cyan]Running algorithms...", total=len(self.ALGORITHM)
                )

                for al in self.ALGORITHM:
                    report = {}
                    alg = self.ALGORITHM[al].fit(X_train, y_train)
                    if al in self.model:
                        alg = self.model[1]
                    progress.console.print(f"{al} is run ...")
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
                    progress.advance(task)
            else:
                title = "Cross-Validation"
                progress.console.print("Using cross-validation ...")

                kfold = KFold(n_splits=5, shuffle=True, random_state=self.dataset.seed)
                task = progress.add_task(
                    "[cyan]Running algorithms...", total=len(self.ALGORITHM)
                )
                for al in self.ALGORITHM:
                    alg = self.ALGORITHM[al]
                    if al in self.model:
                        alg = self.model[1]
                    print(f"{al} is run ...")
                    report = Regression.__cross_val(
                        metrics=self.METRICS,
                        estimator=alg,
                        X=X_train,
                        y=y_train,
                        cv=kfold,
                    )
                    result[al] = report
                    progress.advance(task)
        self.result_compare_models = result

        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "table":
            console.print()
            cols = [""] + list(result["Linear Regression"].keys())
            table = Table(*cols, title=title, box=box.HORIZONTALS)
            for i, v in result.items():
                v = [f"{x:.4f}" for x in v.values()]
                res = [i] + v
                if i in self.model:
                    res = [Text(c, style="bold magenta") for c in res]
                table.add_row(*res)
            console.print(table)
        elif output == "only_score":
            rest = {}
            for i in result:
                if title == "Train-Validation":
                    rest[i] = round(result[i]["mape_val"], 2)
                else:
                    rest[i] = round(result[i]["mean_absolute_percentage_error"], 2)
            return rest
        else:
            raise result
