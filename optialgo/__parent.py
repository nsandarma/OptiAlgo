from sklearn.model_selection import (
    GridSearchCV,
)
import numpy as np
import pickle
from abc import ABC, abstractmethod
import warnings

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

        Args:
            param_grid : A dictionary where the keys are parameter names and the values are lists of parameter settings to try. This dictionary is used for performing grid search.
            inplace : If True, the best parameters found by grid search will be set to the model in place. Default is False.

        Returns:
            tuple or None:
                If `inplace` is False, returns a tuple containing the best score and the best parameters
                found by grid search.
                If `inplace` is True, the method sets the best parameters to the model and returns None.

        Raises:
            ValueError: If the `self.model` attribute is not found.

        Examples:
        ```python
        reg = Regression(dataset, algorithm='linear_regression') # or Classification
        param_grid = {'alpha': [0.1, 0.01, 0.001], 'max_iter': [100, 1000, 10000]}
        best_score, best_params = reg.find_best_params(param_grid)
        print("Best Score:", best_score)
        print("Best Parameters:", best_params)
        ```
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

        Args:
            params : A dictionary containing the parameter names and values to be set for the model. The keys should be the names of the parameters, and the values should be the desired values for those parameters.

        Examples:
        ```python
        reg = Regession(dataset, algorithm='linear_regression')
        new_params = {'alpha': 0.1, 'max_iter': 1000}
        reg.set_params(new_params)
        print(reg.model[1].get_params())
        # output : {'alpha': 0.1, 'copy_X': True, 'fit_intercept': True, 'max_iter': 1000, 'normalize': 'deprecated', ...}
        ```
        """
        self.model[1].set_params(**params)
        self.__model = (self.model[0], self.model[1])
        return

    # <---- Tuning

    # ----> Modelling
    def set_model(self, algorithm: str):
        """
        Determine the algorithm to use

        Args:
            algorithm
        """
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

    def predict(self, X_test: np.ndarray, transform: bool = True):
        """
        Predict for the given test data.

        Args:
            X_test : The test data to predict labels for.
            transform : Whether to transform the test data using `flow_from_array` method before prediction (default is True).

        Returns:
            The predicted for the test data.

        Raises:
            ValueError: If the model is not found.
        """
        if not self.model:
            raise ValueError("model not found !")
        X_test = self.dataset.flow_from_array(X_test) if transform else X_test
        pred = self.model[1].predict(X_test)
        return self.dataset.get_label(pred)

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
        """
        Gets the dataset.

        Returns:
            The dataset.
        """
        return self.__dataset

    @property
    def get_params_from_model(self):
        """
        Gets parameters from the model.

        Raises:
            ValueError: If the model is not found.

        Returns:
            dict: Parameters of the model.
        """

        if not self.model:
            raise ValueError("model not found !")
        return self.model[1].get_params()

    @property
    def get_result_compare_models(self):
        """
        Gets the result of model comparison.

        Raises:
            AttributeError: If the result_compare_models attribute is not found.

        Returns:
            The result of model comparison.
        """
        self.not_found("result_compare_models")
        return self.result_compare_models

    @property
    def model(self):
        """Gets the model.

        Returns:
            The model.
        """
        return self.__model

    @property
    def get_metrics(self):
        """
        Gets the metrics.

        Returns:
            list: A list of metrics.
        """
        return self.METRICS

    @property
    def get_list_models(self):
        """
        Gets the list of models.

        Returns:
            list: A list of model names.
        """
        return list(self.ALGORITHM.keys())
