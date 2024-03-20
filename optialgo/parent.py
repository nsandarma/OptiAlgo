from sklearn.model_selection import (
    GridSearchCV,
    train_test_split,
    StratifiedKFold,
)
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, TargetEncoder
import pandas as pd
import pickle
from abc import ABC, abstractmethod

import warnings

warnings.filterwarnings("always")


class Parent(ABC):
    @abstractmethod
    def __str__(self) -> str: ...

    @abstractmethod
    def compare_model(self): ...

    @abstractmethod
    def score(self, y_true, y_pred): ...

    # ---> Preprocessing Data

    def pca(self, n_components, X):
        pc = PCA(n_components=n_components)
        pc.fit(X)
        return pc.transform(X)

    def check_imbalance(dataset: pd.DataFrame, target_column):
        class_distribution = dataset[target_column].value_counts(normalize=True)
        class_minority = class_distribution[
            class_distribution == class_distribution.min()
        ]

        imbalance_threshold = 0.02

        status = False

        if class_distribution.var() >= imbalance_threshold:
            imbalance_info = f"""
            The {class_minority.index.tolist()} class has an imbalance of {class_minority.values}
            \nConsider handling class imbalance. 
            """
            print(imbalance_info)
            status = True
        return status

    def handling_missing_values(
        self, data: pd.DataFrame, imputation=None, inplace=False
    ) -> pd.DataFrame:
        """
        Handles missing values in the dataset.

        Parameters:
            data (DataFrame): The dataset containing missing values.
            imputation (str or dict, optional): Specifies the method for handling missing values.
                                                If None, missing values are imputed based on the data type:
                                                    - For object dtype: mode
                                                    - For numeric dtypes: median
                                                If 'drop', rows with missing values are dropped.
                                                If a dictionary is provided, missing values are imputed based on the dictionary.
                                                Default is None.
            inplace (bool, optional): Whether to modify the DataFrame in place. Default is False.

        Returns:
            DataFrame: The DataFrame with missing values handled.

        Raises:
            ValueError: If imputation method is not recognized.

        Note:
            This method handles missing values in the dataset based on the specified imputation method.
            If inplace is True, the method modifies the original DataFrame.
        """
        miss_value = data.isna().sum().to_dict()
        miss_value = {
            column: value for column, value in miss_value.items() if value != 0
        }
        data = data.copy()
        if imputation is None:
            for i in miss_value:
                if data[i].dtype == "object":
                    val = data[i].mode()[0]
                    data[i] = data[i].fillna(val)
                elif data[i].dtype in ["int64", "int32", "float64", "float32"]:
                    val = data[i].median()
                    data[i] = data[i].fillna(val)
                else:
                    raise ValueError("Not Defined")
        elif imputation == "drop":
            data = data.dropna()
        else:
            for i in miss_value:
                data[i] = data.fillna(imputation[i])
        if inplace:
            self.data = data
            return self
        else:
            return data

    def encoding(data: pd.DataFrame, features_cat: str, target: str):
        target_encoder = TargetEncoder().fit(data[features_cat], data[target])
        X = target_encoder.transform(data[features_cat])
        for i, v in enumerate(features_cat):
            data[v] = X[:, i]
        data = data.drop(columns=[target])
        return data, target_encoder

    def encoding_predict(encoder, X_test, features_cat):
        X = encoder.transform(X_test[features_cat])
        for i, v in enumerate(features_cat):
            X_test[v] = X[:, 1]
        return X_test

    def decoding(data: pd.DataFrame, data_categories):
        for i in data_categories.columns:
            data[i] = data_categories[i]
        return data

    def check_col_categories(self, data: pd.DataFrame):
        i = []
        for j in data.columns:
            if data[j].dtype == "object":
                i.append(j)
        return i

    def split_data(self, train_size) -> np.ndarray:
        """
        Splits the data into training and testing sets.

        Parameters:
            train_size (float): The proportion of the dataset to include in the training split.
                                Should be between 0.0 and 1.0.

        Returns:
            tuple: A tuple containing four elements: X_train, X_test, y_train, and y_test.
                   X_train (array-like): The training input samples.
                   X_test (array-like): The testing input samples.
                   y_train (array-like): The training target values.
                   y_test (array-like): The testing target values.

        Note:
            If the model type is 'Classification', the data will be stratified based on the target variable y,
            ensuring that the distribution of classes in the training and testing sets remains similar.
            If the model type is not 'Classification', the stratify parameter will be set to None.
        """
        stratify = self.y if self.model_type == "Classification" else None
        return train_test_split(
            self.X,
            self.y,
            random_state=self.seed,
            train_size=train_size,
            stratify=stratify,
        )

    # Preprocessing Data <----

    def fit(self, data: pd.DataFrame, target: str, features: list, norm=True, seed=42):
        """
        Fits the model to the provided data.

        Parameters:
            data (DataFrame): The dataset containing the features and target variable.
            target (str): The name of the target variable.
            features (list): A list of feature names.
            norm (bool, optional): Whether to normalize the features. Default is True.
            seed (int, optional): The random seed for reproducibility. Default is 42.

        Returns:
            self: The modified instance of the class.

        Raises:
            ValueError: If missing values are found in the dataset.
                        If the target variable is of object type.

        Note:
            This method prepares the data for modeling by handling missing values, encoding categorical columns,
            encoding the target variable (if applicable), and normalizing features (if specified).
        """
        data = data.copy()
        self.data = data[features]
        self.features = features
        self.seed = seed

        # Missing Values Handler
        if sum(data.isna().sum().values) > 0:
            miss_value = {
                column: value
                for column, value in data.isna().sum().items()
                if value != 0
            }
            raise ValueError(f"Missing Value in {miss_value}")

        # Check Imbalance
        if self.model_type == "Classification":
            self.status_imbalance = Parent.check_imbalance(data, target)

        # Encoding Columns
        if any(self.check_col_categories(data[features])):
            data_categories = data[features].select_dtypes("object")
            cols_categories = data_categories.columns.tolist()
            X, encoder = Parent.encoding(
                data=data, features_cat=cols_categories, target=target
            )
            self.X_encoder = encoder
            self.cols_encoded = cols_categories
            self.data_categories = data_categories
        X = data[features]

        y = data[target].values
        if data[target].dtype == object:
            labelencoder = LabelEncoder().fit(y)
            y = labelencoder.transform(y)
            self.y_encoder = labelencoder

        # Features Norm
        if norm:
            scaler = MinMaxScaler()
            scaler.fit(X.values)
            X = scaler.transform(X.values)
            self.scaler = scaler
        else:
            X = X.values

        self.X = X
        self.y = y
        self.norm = norm

        return self

    # ----> Modelling

    def set_model(self, algo_name, X_train=None, y_train=None):
        """
        Sets the model with the specified algorithm and training data.

        Parameters:
            algo_name (str): The name of the algorithm to use for modeling.
            X_train (array-like, optional): The training input samples. Default is None.
            y_train (array-like, optional): The training target values. Default is None.
        Returns:
            self: The modified instance of the class.
        Raises:
            ValueError: If the dimensions of X_train or y_train do not match the expected dimensions.
                        If the algorithm name is not found in the list of supported algorithms.
        Note:
            If X_train and y_train are not provided, the method uses the original dataset (self.X and self.y).
            The algorithm must be available in the dictionary ALGORITHM, defined within the class.
        """
        if isinstance(X_train, np.ndarray) and isinstance(X_train, np.ndarray):
            pass
        else:
            X_train = self.X
            y_train = self.y

        if (X_train.shape[1] != self.X.shape[1]) or (y_train.ndim != self.y.ndim):
            raise ValueError(
                "Dimension of X_train or y_train does not match the expected dimension."
            )

        self.X_train = X_train
        self.y_train = y_train
        try:
            if algo_name in self.ALGORITHM:
                model_instance = self.ALGORITHM[algo_name]
                model = model_instance.fit(X_train, y_train)
                self.model = (algo_name, model)
                return self
            else:
                raise ValueError("Algorithm not found in the list.")
        except Exception as e:
            print("Error:", e)
            return self

    def grid_search_cv(self, param_grid, algo_name=None):
        """
        Performs grid search cross-validation to tune hyperparameters for the specified algorithm.

        Parameters:
            param_grid (dict): The parameter grid to search over.
            algo_name (str, optional): The name of the algorithm. If None, it uses the current model.

        Returns:
            GridSearchCV: A GridSearchCV object fitted with the specified parameters.

        Raises:
            ValueError: If the specified algorithm is not found in the list.

        Note:
            This method performs grid search cross-validation to tune hyperparameters for the specified algorithm.
            If algo_name is None, it uses the current model. Otherwise, it uses the specified algorithm.
        """
        if algo_name == None:
            self.not_found("model")
            alg = self.model[1]
        else:
            if algo_name not in self.ALGORITHM.keys():
                raise ValueError("Algorithm not found in the list.")
            alg = self.ALGORITHM[alg_name]

        return GridSearchCV(alg, param_grid).fit(self.X, self.y)

    def find_best_params(
        self, param_grid, algo_name=None, X_train=None, y_train=None, n_splits=5
    ) -> dict:
        """
        Finds the best hyperparameters for the specified algorithm using grid search.

        Parameters:
            param_grid (dict): The parameter grid to search over.
            algo_name (str, optional): The name of the algorithm. If None, it uses the current model.
            X_train (array-like, optional): The feature matrix for training. Default is None.
            y_train (array-like, optional): The target vector for training. Default is None.
            n_splits (int, optional): Number of splits for cross-validation. Default is 5.

        Returns:
            dict: A dictionary containing the best parameters found (best_params) and the corresponding best score (score).

        Raises:
            ValueError: If the specified algorithm is not found in the list.

        Note:
            This method performs grid search to find the best hyperparameters for the specified algorithm.
            It uses stratified k-fold cross-validation for parameter tuning.
        """

        if algo_name == None:
            self.not_found("model")
            alg = self.model[1]
        else:
            if algo_name not in self.ALGORITHM.keys():
                raise ValueError("Algorithm not found in the list.")
            alg = self.ALGORITHM[alg_name]

        if X_train == None and y_train == None:
            X_train = self.X
            y_train = self.y

        if self.model_type == "Classification":
            kfold = StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=self.seed
            )
            score = "accuracy"
        else:
            kfold = None
            score = "neg_mean_absolute_percentage_error"

        clf = GridSearchCV(
            estimator=alg, param_grid=param_grid, cv=kfold, scoring=score
        )
        clf.fit(X_train, y_train)
        score = clf.best_score_
        if self.model_type == "Regression":
            score = abs(clf.best_score_)
        res = {"best_params": clf.best_params_, "score": score}
        return res

    def tuning(self, params):
        """
        Tunes the hyperparameters of the model.

        Parameters:
            params (dict): A dictionary containing the hyperparameters to be tuned.
        Returns:
            self: The modified instance of the class.
        Raises:
            ValueError: If the model algorithm is not found.
                        If a parameter provided for tuning is not required for the model.
        Note:
            This method tunes the hyperparameters of the existing model.
            The hyperparameters to be tuned must be provided in the params dictionary.
        """
        if not hasattr(self, "model"):
            raise ValueError("Model algorithm not found")

        algo_name = self.model[0]

        model_instance = self.model[1]

        params_required = list(model_instance.get_params().keys())

        for key in params:
            if key not in params_required:
                raise ValueError(
                    "Parameter '{}' is not required for this model.".format(key)
                )

        model = self.model[1]
        model = model.set_params(**params)
        self.model = algo_name, model
        return self

    def predict(self, X_test: pd.DataFrame, output=None):
        """
        Predicts the target values for the given test data.

        Parameters:
            X_test (DataFrame or ndarray): The test data for prediction.
            output (str, optional): Specifies the format of the output. Default is None.
                                    If set to 'dataframe', returns the predictions along with the test data.

        Returns:
            array-like or DataFrame: Predicted target values.

        Raises:
            NotImplementedError: If the model is not defined.
            ValueError: If the number of features in the test data is not equal to the number of features in the training data.
                        If the data type of X_test is not 'ndarray' or 'DataFrame'.

        Note:
            This method predicts the target values for the given test data using the trained model.
            If output is set to 'dataframe', the method returns a DataFrame containing the test data and predicted values.
        """
        if not hasattr(self, "model"):
            raise NotImplementedError("Model Not Define")
        if X_test.shape[1] != self.X_train.shape[1]:
            raise ValueError(
                "The number of features in the test data is not equal to the number of features in the training data."
            )

        if type(X_test).__name__ not in ["ndarray", "DataFrame"]:
            raise ValueError("X_test data type must be ndarray or dataframe")

        X = X_test
        if isinstance(X_test, pd.DataFrame):
            if hasattr(self, "cols_encoded"):
                X = Parent.encoding_predict(self.X_encoder, X, self.cols_encoded)
            X = X.values
        else:
            if X_test.dtype == "object":
                X = pd.DataFrame(data=X, columns=self.features)
                X = Parent.encoding_predict(self.X_encoder, X, self.cols_encoded)
                X = X.values
            X_test = pd.DataFrame(data=X, columns=self.features)

        if hasattr(self, "scaler"):
            if (X.min() >= 0) and (X.max() <= 1):
                ...
            else:
                X = self.scaler.transform(X)

        pred = self.model[1].predict(X)

        if hasattr(self, "y_encoder"):
            pred = self.y_encoder.inverse_transform(pred)
        if output == "dataframe":
            X_test["pred"] = pred
            return X_test
        return pred

    def predict_cli(self, output="dict") -> None:
        """
        Predicts target values based on user input through the command-line interface (CLI).

        Returns:
            None

        Raises:
            ValueError: If the input value is not within the categorical features' categories.

        Note:
            This method allows the user to input feature values through the command line and predicts the target values
            using the trained model. It prints the input features along with the predicted target values.
        """
        input_ = {}
        if hasattr(self, "cols_encoded"):
            cols_encoded = self.cols_encoded
            ind = 0
            cats = self.X_encoder.categories_
            for i in self.features:
                if i in cols_encoded:
                    category = cats[ind].tolist()
                    v = input(f"{i} {category}\t: ")
                    if v not in category:
                        raise ValueError(f"{v} not in category")
                    ind += 1
                else:
                    v = input(f"{i}\t\t: ")
                input_[i] = [v]
            rest = pd.DataFrame(input_)
            X = Parent.encoding_predict(
                self.X_encoder, rest, features_cat=cols_encoded
            ).values
        else:
            for i in self.features:
                v = input(f"{i} \t: ")
            input_[i] = [v]
            rest = pd.DataFrame(input_)
            X = rest.values
        pred = self.model[1].predict(X)
        if hasattr(self, "y_encoder"):
            pred = self.y_encoder.inverse_transform(pred)
        rest["Predictions"] = pred
        print()
        print("-------> RESULT <---------")
        if output == "dataframe":
            print(rest)
        elif output == "only_pred":
            print(rest["Predictions"].values[0])
        else:
            print(rest.to_dict(orient="records")[0])

    def save_model(self):
        """
        Serialize and save the model object using pickle.

        Returns:
            bytes: Serialized representation of the model object.

        Raises:
            PickleError: If serialization fails.
        """

        return pickle.dumps(self)

    # Modelling <----

    def not_found(self, attr: str):
        if not hasattr(self, attr):
            raise ValueError(f"{attr} not found")

    # Getter
    @property
    def get_X(self):
        return self.X

    @property
    def get_y(self):
        return self.y

    @property
    def get_X_train(self):
        return self.X_train

    @property
    def get_X_test(self):
        return self.X_test

    @property
    def get_params_from_model(self):
        self.not_found("model")
        return self.model[1].get_params()

    @property
    def get_result_compare_models(self):
        self.not_found("result_compare_models")
        return self.result_compare_models

    @property
    def get_metrics(self):
        return self.METRICS

    @property
    def get_algorithm(self):
        return self.ALGORITHM

    @property
    def get_list_models(self):
        return list(self.ALGORITHM.keys())
