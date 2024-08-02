import pandas as pd
import numpy as np
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
from typing import Literal, Optional

from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# optialgo
from . import Parent
from .dataset import Dataset

import warnings

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box
from rich.progress import Progress


warnings.filterwarnings("always")


class Classification(Parent):
    ALGORITHM = {
        "Naive Bayes": MultinomialNB(),
        "K-Nearest Neighbor": KNeighborsClassifier(),
        "SVM": SVC(),
        "Logistic Regression": LogisticRegression(max_iter=4000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "XGBoost": XGBClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(),
    }
    METRICS = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
    }
    model_type = "Classification"

    def __init__(
        self,
        dataset: Dataset,
        algorithm: Optional[
            Literal[
                "Naive Bayes",
                "K-Nearest Neighbor",
                "SVM",
                "Logistic Regression",
                "Random Forest",
                "Decision Tree",
                "XGBoost",
                "Gradient Boosting",
            ]
        ] = None,
    ):
        """
        Initializes the class with the provided dataset and algorithm.

        If the dataset represents a binary classification problem,
        it adds the AUC metric to the classification metrics.

        Args:
            dataset : An instance of the Dataset class, which includes the data and its characteristics.
            algorithm : The algorithm to be used for classification. Default is None.

        Examples:
        ```python
        dataset = Dataset(dataframe, norm=True, test_size=0.3, seed=123)
        dataset.fit(features=features,target=target)
        classifier = Classifier(dataset, algorithm='random_forest')
        ```
        """
        if dataset.class_type == "binary":
            Classification.METRICS["AUC"] = roc_auc_score
        return super().__init__(dataset, algorithm)

    def __str__(self) -> str:
        return "<Classificaton Object>"

    def __cross_val(metrics, X, y, estimator, cv, class_type):
        c = cross_validate(estimator, X, y, scoring="accuracy", cv=cv)
        c["fit_time"] = c["fit_time"].mean()
        c["score_time"] = c["score_time"].mean()
        c["test_score"] = c["test_score"].mean()
        c["accuracy"] = c.pop("test_score")
        c["recall"] = cross_val_score(estimator, X, y, scoring="recall_micro").mean()
        c["precision"] = cross_val_score(
            estimator, X, y, scoring="precision_micro"
        ).mean()
        c["f1"] = cross_val_score(estimator, X, y, scoring="f1_micro").mean()
        if class_type == "binary":
            c["precision"] = cross_val_score(
                estimator, X, y, scoring="precision"
            ).mean()
            c["recall"] = cross_val_score(estimator, X, y, scoring="recall").mean()
            c["f1"] = cross_val_score(estimator, X, y, scoring="f1").mean()
            c["auc"] = cross_val_score(estimator, X, y, scoring="roc_auc").mean()
            if type(estimator).__name__ != "SVC":
                c["log_loss"] = cross_val_score(
                    estimator, X, y, scoring="neg_log_loss"
                ).mean()
        return c

    def score(self, y_true, y_pred, metric="accuracy"):
        """
        Calculates the score based on the predicted and true target values.

        Args:
            y_true (array-like): The true target values.
            y_pred (array-like): The predicted target values.
            metric (str, optional): The scoring metric to use. Default is 'accuracy'.
                                    Other options depend on the problem type:
                                    - For multiclass classification: 'precision', 'recall', 'f1', 'accuracy'
                                    - For binary classification or regression: 'accuracy', 'precision', 'recall', 'f1', 'r2'
                                    See sklearn.metrics for available metrics.

        Returns:
            float: The score calculated based on the specified metric.

        Note:
            This method calculates the score based on the predicted and true target values.
            For multiclass classification, it uses weighted averaging for precision, recall, and F1 score.
            For other problem types, it returns the specified metric score directly.
        """
        average = "weighted" if self.dataset.class_type == "multiclass" else None
        if metric == "accuracy":
            return self.METRICS[metric](y_true, y_pred)
        if self.dataset.class_type == "multiclass":
            return self.METRICS[metric](
                y_true, y_pred, average=average, zero_division=0.0
            )
        else:
            return self.METRICS[metric](y_true, y_pred)

    def score_report(self, y_true, y_pred):
        """
        Generates a report containing various metric scores and classification report between true and predicted target values.

        Args:
            y_true (array-like): The true target values.
            y_pred (array-like): The predicted target values.

        Returns:
            dict: A dictionary containing metric scores calculated for each metric defined in the METRICS dictionary
                  along with the classification report.

        Note:
            This method generates a report containing various metric scores and classification report
            between true and predicted target values. It calculates metric scores for each metric defined in the METRICS dictionary
            and includes the classification report generated using scikit-learn's classification_report function.
        """
        res = {}
        for i in self.METRICS:
            res[i] = self.score(y_true, y_pred, metric=i)
        res["classification_report"] = classification_report(
            y_true=y_true, y_pred=y_pred, zero_division=0.0, output_dict=True
        )
        return res

    def compare_model(
        self,
        output: Literal["dict", "dataframe", "table", "only_score"] = "dict",
        train_val: bool = True,
        n_splits: int = 5,
        verbose: bool = True,
    ):
        """
        Compares multiple classification models based on various metrics and returns the results.

        This function runs a set of algorithms on the training data, evaluates them on the training and
        validation sets or using cross-validation, and compiles the performance metrics. The results
        can be output in different formats, including dictionary, pandas DataFrame, or a formatted table.

        Args:
            output :
                The format of the output. It can be:
                - "dict": Returns the results as a dictionary (default).
                - "dataframe": Returns the results as a pandas DataFrame.
                - "table": Prints the results as a formatted table.
                - "only_score": Returns only the accuracy scores in a simplified dictionary.

        train_val : If True, the function evaluates the models using a train-validation split.
            If False, the function uses cross-validation. Default is True.

        Returns:
            dict or pd.DataFrame or None

        Raises:
            ValueError : If the task type is not "classification" or "regression".

        Examples:
        ```python
        dataset = Dataset(dataframe, norm=True, test_size=0.3, seed=123)
        dataset.fit(features=features,target=target,t="classification")
        classifier = Classification(dataset, algorithm='random_forest')
        results = classifier.compare_model(output='dataframe', train_val=True)
        print(results)

        ```
        """
        result = {}
        X_train, X_test, y_train, y_test = self.dataset.get_x_y()

        console = Console()

        with Progress() as progress:
            if verbose:
                task = progress.add_task(
                    "[cyan]Running algorithms...", total=len(self.ALGORITHM)
                )
            clf_report = {}
            for al in self.ALGORITHM:
                report = {}
                alg = self.model[1] if al in self.model else self.ALGORITHM[al]
                if verbose:
                    progress.advance(task)
                if train_val:
                    title = "Train-Validation"
                    alg = alg.fit(X_train, y_train)
                    pred_train = alg.predict(X_train)
                    pred_val = alg.predict(X_test)
                    report["accuracy_train"] = self.score(y_train, pred_train)
                    report["accuracy_val"] = self.score(y_test, pred_val)
                    report["precision_train"] = self.score(
                        y_train, pred_train, metric="precision"
                    )
                    report["precision_val"] = self.score(
                        y_test, pred_val, metric="precision"
                    )
                    report["recall_train"] = self.score(
                        y_train, pred_train, metric="recall"
                    )
                    report["recall_val"] = self.score(y_test, pred_val, metric="recall")
                    report["f1_train"] = self.score(y_train, pred_train, metric="f1")
                    report["f1_val"] = self.score(y_test, pred_val, metric="f1")
                    if self.dataset.class_type == "binary":
                        report["auc_train"] = self.score(
                            y_train, pred_train, metric="AUC"
                        )
                        report["auc_val"] = self.score(y_test, pred_val)

                    result[al] = report
                    clf_report[al] = classification_report(
                        y_test, pred_val, labels=np.unique(pred_val)
                    )
                else:
                    title = f"Cross-Validation (n_splits: {n_splits})"
                    # progress.console.print("Using cross-validation ...")
                    kfold = StratifiedKFold(
                        n_splits=n_splits, shuffle=True, random_state=self.dataset.seed
                    )
                    report = Classification.__cross_val(
                        metrics=self.METRICS,
                        estimator=alg,
                        X=X_train,
                        y=y_train,
                        cv=kfold,
                        class_type=self.dataset.class_type,
                    )
                    result[al] = report

        if clf_report:
            self.result_compare_models = clf_report
        else:
            self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "table":
            cols = [""] + list(result["Naive Bayes"].keys())
            table = Table(*cols, title=title, box=box.HORIZONTALS)
            for i, v in result.items():
                v = [f"{x:.4f}" for x in v.values()]
                res = [i] + v
                if i in self.model:
                    res = [Text(c, style="bold magenta") for c in res]
                table.add_row(*res)
            console.print(table)
        elif output == "only_score":
            if title == "Train-Validation":
                rest = {i: round(result[i]["accuracy_val"], 2) for i in result}
            else:
                rest = {i: round(result[i]["accuracy"], 2) for i in result}
            return rest
        else:
            return result
