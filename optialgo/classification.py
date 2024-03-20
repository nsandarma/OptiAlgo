import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    KFold,
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
from sklearn.preprocessing import MinMaxScaler

# Import Algorithm Module from sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from collections import defaultdict
from .parent import Parent

# Sampling
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler

import warnings

warnings.filterwarnings("always")

ALGORITHM_NAMES = [
    "Naive Bayes",
    "K-Nearest Neighbor",
    "SVM",
    "Logistic Regression",
    "Random Forest",
    "Decision Tree Classifier",
    "XGBoost",
    "Gradient Boosting",
]

ALGORITHM_OBJECT = [
    MultinomialNB(),
    KNeighborsClassifier(),
    SVC(),
    LogisticRegression(max_iter=3000),
    RandomForestClassifier(),
    DecisionTreeClassifier(),
    XGBClassifier(),
    GradientBoostingClassifier(),
]
ALGORITHM_CLF = dict(zip(ALGORITHM_NAMES, ALGORITHM_OBJECT))
METRICS_NAMES = ["accuracy", "precision", "recall", "f1"]
METRICS_OBJECT = [accuracy_score, precision_score, recall_score, f1_score]


class Classification(Parent):
    ALGORITHM = ALGORITHM_CLF
    METRICS = dict(zip(METRICS_NAMES, METRICS_OBJECT))
    model_type = "Classification"

    def fit(self, data: pd.DataFrame, target: str, features: list, norm=True):
        len_class = len(data[target].unique())
        self.class_type = "binary" if len_class == 2 else "multiclass"
        if self.class_type == "binary":
            Classification.METRICS["AUC"] = roc_auc_score
        return super().fit(data, target, features, norm)

    def __str__(self) -> str:
        return "<Classificaton Object>"

    def cross_val(metrics, X, y, estimator, cv, class_type):
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

        Parameters:
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
        average = "weighted" if self.class_type == "multiclass" else None
        if metric == "accuracy":
            return self.METRICS[metric](y_true, y_pred)
        if self.class_type == "multiclass":
            return self.METRICS[metric](
                y_true, y_pred, average=average, zero_division=0.0
            )
        else:
            return self.METRICS[metric](y_true, y_pred)

    def sampling(self, method, X, y, sampling_strategy="auto"):
        """
        Performs sampling on the dataset using the specified method.

        Parameters:
            method (str): The sampling method to use. Supported options: 'under', 'over', 'SMOTE'.
            X (array-like): The feature matrix.
            y (array-like): The target vector.
            sampling_strategy (str or float or dict, optional): The sampling strategy to apply. Default is 'auto'.
                                                                Refer to the documentation of imbalanced-learn
                                                                library for available options.

        Returns:
            tuple: A tuple containing the resampled feature matrix (X_resampled) and the resampled target vector (y_resampled).

        Raises:
            ValueError: If the specified method is not found.

        Note:
            This method performs sampling on the dataset to address class imbalance.
            Supported sampling methods include random undersampling ('under'), random oversampling ('over'),
            and Synthetic Minority Over-sampling Technique (SMOTE).
        """
        random_state = self.seed
        if method == "under":
            sampler = RandomUnderSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        elif method == "over":
            sampler = RandomOverSampler(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        elif method == "SMOTE":
            sampler = SMOTE(
                sampling_strategy=sampling_strategy, random_state=random_state
            )
        else:
            raise ValueError("method not found !")
        return sampler.fit_resample(X, y)

    def score_report(self, y_true, y_pred):
        """
        Generates a report containing various metric scores and classification report between true and predicted target values.

        Parameters:
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
        X_train=None,
        X_test=None,
        y_train=None,
        y_test=None,
        output="dict",
        train_val=False,
    ):
        """
        Compares the performance of different models using either train-test split or cross-validation.

        Parameters:
            X_train (array-like, optional): The feature matrix for training. Default is None.
            X_test (array-like, optional): The feature matrix for testing. Default is None.
            y_train (array-like, optional): The target vector for training. Default is None.
            y_test (array-like, optional): The target vector for testing. Default is None.
            output (str, optional): Specifies the format of the output. Default is 'dict'.
                                    Options: 'dict' (dictionary), 'dataframe' (DataFrame), 'only_accuracy'.
            train_val (bool, optional): Whether to compute performance metrics on both training and validation sets.
                                        Applicable only if X_train, X_test, y_train, and y_test are provided.
                                        Default is False.

        Returns:
            dict or DataFrame: A dictionary or DataFrame containing the performance metrics of the models.

        Note:
            This method compares the performance of different models using either train-test split or cross-validation.
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
                    report["accuracy_train"] = self.score(y_train, pred_train)
                    report["acc_val"] = self.score(y_test, pred_val)
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
                    if self.class_type == "binary":
                        report["auc_train"] = self.score(
                            y_train, pred_train, metric="AUC"
                        )
                        report["auc_val"] = self.score(y_test, pred_val)
                    result[al] = report
            else:
                for al in self.ALGORITHM:
                    alg = self.ALGORITHM[al].fit(X_train, y_train)
                    print(f"{al} is run ...")
                    y_pred = alg.predict(X_test)
                    report = self.score_report(y_test, y_pred)
                    result[al] = report
        else:
            kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
            for al in self.ALGORITHM:
                alg = self.ALGORITHM[al]
                print(f"{al} is run ...")
                report = Classification.cross_val(
                    metrics=self.METRICS,
                    estimator=alg,
                    X=self.X,
                    y=self.y,
                    cv=kfold,
                    class_type=self.class_type,
                )
                result[al] = report
        self.result_compare_models = result
        if output == "dataframe":
            return pd.DataFrame.from_dict(result, orient="index")
        elif output == "only_accuracy":
            rest = {}
            for i in result:
                rest[i] = round(result[i]["accuracy"], 2)
            return rest
        else:
            return result
