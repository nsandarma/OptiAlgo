from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import pandas as pd
import pickle
from abc import ABC, abstractmethod
import warnings

from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich import box
from rich.text import Text

warnings.filterwarnings("always")


class Parent(ABC):
    from .dataset import Dataset

    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def compare_model(self): ...

    @abstractmethod
    def score(self, y_true, y_pred): ...

    def __init__(self, dataset: Dataset, algorithm: str = None):
        from . import Dataset, TextDataset

        if not isinstance(dataset, (Dataset, TextDataset)):
            raise TypeError(
                f"Expected object of type {dataset.__name__}, got {type(dataset).__name__} instead."
            )
        if not hasattr(dataset, "train"):
            raise NotImplementedError("dataset not implemented fit !")
        self.__dataset = dataset
        self.set_model(algorithm)

    # ----> Tuning
    def find_best_params(self, param_grid: dict, inplace=False):
        """
        Perform hyperparameter tuning to find the best parameters for the model.

        This method uses grid search cross-validation to find the best parameters for the model
        stored in the `self.model` attribute. It takes a dictionary of parameter values to search
        over and returns the best score and parameters found. Optionally, it can set the best
        parameters to the model in place.

        Parameters
        ----------
        param_grid : dict
            A dictionary where the keys are parameter names and the values are lists of parameter
            settings to try. This dictionary is used for performing grid search.

        inplace : bool, optional
            If True, the best parameters found by grid search will be set to the model in place.
            Default is False.

        Returns
        -------
        tuple or None
            If `inplace` is False, returns a tuple containing the best score and the best parameters
            found by grid search.
            If `inplace` is True, the method sets the best parameters to the model and returns None.

        Raises
        ------
        ValueError
            If the `self.model` attribute is not found.

        Example
        -------
        >>> reg = Regression(dataset, algorithm='linear_regression')
        >>> param_grid = {'alpha': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}
        >>> best_score, best_params = reg.find_best_params(param_grid)
        >>> print("Best Score:", best_score)
        >>> print("Best Parameters:", best_params)
        """
        if not self.model:
            raise ValueError("model not found !")
        X_train, _, y_train, _ = self.dataset.get_x_y()

        res = GridSearchCV(estimator=self.model[1], param_grid=param_grid).fit(
            X_train, y_train
        )

        if inplace:
            self.set_params(res.best_params_)
            return
        return res.best_score_, res.best_params_

    def set_params(self, params: dict) -> None:
        """
        Set parameters for the model.

        This method updates the parameters of the model stored in the `self.model` attribute.
        It uses the provided dictionary of parameters to set new values for the model's parameters.

        Parameters
        ----------
        params : dict
            A dictionary containing the parameter names and values to be set for the model.
            The keys should be the names of the parameters, and the values should be the desired
            values for those parameters.

        Returns
        -------
        None
            This method does not return anything. It updates the model's parameters in place.

        Example
        -------
        >>> reg = Regession(dataset, algorithm='linear_regression')
        >>> new_params = {'alpha': 0.1, 'max_iter': 1000}
        >>> reg.set_params(new_params)
        >>> print(reg.model[1].get_params())
        {'alpha': 0.1, 'copy_X': True, 'fit_intercept': True, 'max_iter': 1000, 'normalize': 'deprecated', ...}
        """
        self.model[1].set_params(**params)
        self.__model = (self.model[0], self.model[1])
        return

    # <---- Tuning

    # ----> Modelling
    def set_model(self, algorithm: str):
        if algorithm:
            if algorithm not in self.get_list_models:
                raise ValueError(f"{algorithm} not found in {self.get_list_models}")
            dataset = self.dataset
            X_train, _, y_train, _ = dataset.get_x_y()
            model = self.ALGORITHM[algorithm].fit(X_train, y_train)
            model = (algorithm, model)
        else:
            model = ()
        self.__model = model

    def predict(self, X_test: np.ndarray, output=None):
        if not self.model:
            raise ValueError("model not found !")
        pred = self.model[1].predict(X_test)
        return pred

    def predict_cli(self, output="dict"):
        if not self.model:
            raise NotImplementedError("model not found !")
        features = self.dataset.feature_names
        console = Console()
        console.print(
            Text("Welcome to the Rich CLI Application!", style="bold underline magenta")
        )
        X = []
        for i in features:
            f = Prompt.ask(f"{i}", default="q")
            if f == "q":
                console.print("Quitting the application. Goodbye!")
                return False
            X.append(f)
        X = self.dataset.flow_from_array(np.array([X]))
        pred = self.predict(X)
        pred = self.dataset.get_label(pred)
        console.print(pred)

    def save(self):
        """
        Serialize and save the model object using pickle.

        Returns:
            bytes: Serialized representation of the optialgo object.

        Raises:
            PickleError: If serialization fails.
        """

        return pickle.dumps(self)

    def save_model(self):
        return pickle.dumps(self.model[1])

    # Modelling <----

    def not_found(self, attr: str):
        if not hasattr(self, attr):
            raise ValueError(f"{attr} not found")

    # Getter
    @property
    def dataset(self):
        return self.__dataset

    @property
    def get_params_from_model(self):
        if not self.model:
            raise ValueError("model not found !")
        return self.model[1].get_params()

    @property
    def get_result_compare_models(self):
        self.not_found("result_compare_models")
        return self.result_compare_models

    @property
    def model(self):
        return self.__model

    @property
    def get_metrics(self):
        return self.METRICS

    @property
    def get_list_models(self):
        return list(self.ALGORITHM.keys())
