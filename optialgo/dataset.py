import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder, OrdinalEncoder
from . import check_imbalance, check_missing_value
from typing import Literal, Optional


class Dataset:
    ENCODERS = {
        "target_mean": TargetEncoder(),
        "one_hot": OneHotEncoder(),
        "ordinal": OrdinalEncoder(),
    }

    def __init__(
        self,
        dataframe: pd.DataFrame,
        norm: bool = False,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        """
        Initializes the Dataset object with the provided dataframe and configuration settings.

        Args:
            dataframe: The input dataframe containing the data to be processed and used for model training.
            norm: A flag indicating whether to normalize the features. Default is False.
            test_size: The proportion of the dataset to include in the test split. Default is 0.2 (20%).
            seed: The random seed for reproducibility of the train-test split. Default is 42.

        Raises:
            ValueError: If there are missing values in the input dataframe, an error is raised with information on which columns contain missing values.

        Examples:
        ```python
        df = pd.read_csv("house_prices.csv")
        dataset = Dataset(dataframe=df)
        ```
        """

        self.__test_size = test_size
        self.__norm = norm
        self.__seed = seed

        # Missing Values Handler
        mv = check_missing_value(dataframe)
        if mv:
            raise ValueError(mv)

        self.__dataframe = dataframe

    def flow_from_dataframe(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the input dataframe using the preprocessing pipeline.

        Args:
            X: The input dataframe to be transformed.

        Returns:
            np.ndarray: The transformed data as a NumPy array.

        Examples:
        ```python
        new_data = pd.DataFrame({"col_a": [1,2,3],"col_b":[1,1,1], "col_c" : [2,2,2]})
        transformed_X = dataset.flow_from_dataframe(new_data)
        print(transformed_X.shape)
        # output : (3,3)
        ```
        """
        return self.pipeline.transform(X)

    def flow_from_array(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the input array using the preprocessing pipeline.

        Args:
            X : The input array to be transformed. It should have the same number of features as the training data and in the same order.

        Returns:
            The transformed data as a NumPy array.

        Examples:
        ```python
        input_array = np.array([[1, 2, 3], [4, 5, 6]])
        transformed_array = dataset.flow_from_array(input_array)
        print(transformed_array.shape)
        # output : (2, 3)
        # Assuming the pipeline transforms it into 3 features
        ```
        """
        X = pd.DataFrame(X, columns=self.__feature_names_in_)
        return self.pipeline.transform(X)

    def get_label(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Converts predicted numerical labels back to their original categorical labels using the label encoder.

        Args:
            y_pred: The predicted numerical labels to be converted.

        Returns:
            The original categorical labels corresponding to the numerical predictions.

        Raises:
            ValueError: If the label encoder has not been fitted or is not available.

        Examples:
        ```python
        y_pred = np.array([0, 1, 2])
        original_labels = dataset.get_label(y_pred)
        print(original_labels)
        # output : array(['class1', 'class2', 'class3'], dtype=object)
        ```

        """
        if not hasattr(self, "label_encoder"):
            raise ValueError("label_encoder not found !")
        return self.label_encoder.inverse_transform(y_pred)

    def get_x_y(self):
        """
        Splits the preprocessed training and testing data into features and target arrays.

        Returns:
            tuple : A tuple containing four elements: X_train, X_test, y_train, y_test

        Examples:
        ```python
        X_train, X_test, y_train, y_test = dataset.get_x_y()
        ```
        """
        X_train = self.__train[self.__features].values
        y_train = self.__train[self.__target].values
        X_test = self.__test[self.__features].values
        y_test = self.__test[self.__target].values
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        features: list,
        target: Optional[str],
        t: Literal["classification", "regression"],
        encoder: dict = None,
        ci=False,
    ):
        """
        Prepares and fits the dataset for a machine learning task by performing necessary preprocessing steps.

        Args:
            features: A list of feature column names to be used for model training.
            target: The name of the target column.
            t: classification or regression
            encoder: A dictionary specifying custom encoders for specific columns. If None, default encoders are used.
            check_imbalance: If True, checks for class imbalance in the target column for classification tasks. If an imbalance is detected, it triggers the imbalance handling procedure.

        Attributes:
            t: The type of machine learning task (`clustering`, `classification`, or `regression`).
            class_type: The classification type (`binary` or `multiclass`) if the task is classification.
            train: The preprocessed training dataset.
            test: The preprocessed testing dataset.
            pipeline: The preprocessing pipeline used for transforming the data.
            features: The list of feature names after preprocessing.
            feature_names: The original feature names before preprocessing.
            target: The target column name.
            label_encoder: The label encoder used for encoding the target column if it is categorical.

        Notes:
            This method performs the following steps:
                ```
                1. Determines the task type (clustering, classification, or regression) based on the target column.
                2. Encodes the target column if it is categorical.
                3. Checks for class imbalance if `ci` is True.
                4. Splits the dataset into training and testing sets.
                5. Applies preprocessing to the features using the specified or default encoders.
                6. Stores the preprocessed training and testing datasets for model training.
                ```

        Examples:
        ```python
        dataset = Dataset(dataframe=df)
        dataset.fit(features=["feature1", "feature2", "feature_3"], target='target_column', t= "classification", check_imbalance=True)

        # if with custom encoder
        encoder = {"one_hot":["feature_1","feature_2"], "target_mean": ["feature_3"]}
        dataset.fit(features=["feature1", "feature2", "feature_3"], target='target_column', t= "classification", check_imbalance=True, encoder= encoder)

        ```
        """

        dataframe = self.__dataframe.copy()

        stratify = None

        if target:
            if t == "classification":
                if ci:
                    check_imbalance(dataframe=dataframe, target=target)
                target_encoder = LabelEncoder().fit(dataframe[target].values)
                dataframe[target] = target_encoder.transform(dataframe[target].values)
                self.__label_encoder = target_encoder
                stratify = dataframe[target]
                self.class_type = (
                    "binary"
                    if len(dataframe[target].value_counts().values) == 2
                    else "multiclass"
                )
            self.__target = target
        else:
            t = "clustering"

        self.t = t

        train, test = Dataset.train_test_split(
            dataframe, test_size=self.test_size, seed=self.seed, stratify=stratify
        )

        pipeline = Dataset.__preprocessing(train, features, target, self.norm, encoder)

        self.__pipeline = pipeline

        feature_names = pipeline[0].get_feature_names_out()
        self.__feature_names_in_ = pipeline[0].feature_names_in_
        self.__features = feature_names

        train_ = pd.DataFrame(pipeline.transform(train), columns=feature_names)
        test_ = pd.DataFrame(pipeline.transform(test), columns=feature_names)

        if target:
            train_[target] = train[target].values
            test_[target] = test[target].values

        self.__train = train_

        self.__test = test_
        return self

    def __encoding(col: list, encoder: str):
        ENCODER_NAMES = list(Dataset.ENCODERS.keys())
        ENCODER = Dataset.ENCODERS

        if encoder not in ENCODER_NAMES:
            raise ValueError("encoder not found ! | [{}]".format(ENCODER_NAMES))
        r = (encoder, ENCODER[encoder], col)
        del ENCODER, ENCODER_NAMES
        return r

    def __preprocessing(
        dataframe: pd.DataFrame, features, target, norm, encoder: dict = None
    ):

        dataframe = dataframe.copy()

        if encoder:
            transformers = []
            for e, c in encoder.items():
                t = Dataset.__encoding(c, e)
                transformers.append(t)
        else:
            cat_feature = dataframe[features].select_dtypes("object").columns.tolist()
            if target:
                cat_transformer = Dataset.__encoding(cat_feature, "target_mean")
            else:
                cat_transformer = Dataset.__encoding(cat_feature, "ordinal")

            transformers = [cat_transformer]

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="passthrough",
            verbose_feature_names_out=False,
        )
        steps = [("preprocessor", preprocessor)]
        if norm:
            steps.append(("scaler", MinMaxScaler()))
        pipeline = Pipeline(steps=steps)
        if target:
            pipeline.fit(dataframe[features], dataframe[target])
        else:
            pipeline.fit(dataframe[features])

        return pipeline

    def train_test_split(dataframe, test_size: float, seed: int, stratify):
        if test_size >= 0.5:
            raise ValueError("test size >= 0.5 !")

        train, test = train_test_split(
            dataframe, test_size=test_size, random_state=seed, stratify=stratify
        )
        return train, test

    def save(self):
        """
        Serialize and save the dataset object using pickle.

        Returns:
            bytes: Serialized representation of the optialgo object.

        Raises:
            PickleError: If serialization fails.
        """

        return pickle.dumps(self)

    def __str__(self):
        if not hasattr(self, "t"):
            return "<Dataset Object>"
        return "<Dataset {} Object>".format(self.t)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        if hasattr(self, "train"):
            X = self.train[self.features].values
            if self.t == "clustering":
                return X[idx]
            y = self.train[self.target].values
            return X[idx], y[idx]
        else:
            return self.dataframe.iloc[idx].values

    # Getter

    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def train(self) -> pd.DataFrame:
        return self.__train

    @property
    def test(self):
        return self.__test

    @property
    def test_size(self) -> float:
        return self.__test_size

    @property
    def norm(self):
        return self.__norm

    @property
    def seed(self):
        return self.__seed

    @property
    def features(self):
        return self.__features

    @features.setter
    def features(self, features: list):
        if len(features) > len(self.train.columns.tolist()):
            raise ValueError("len(features) > len(train.columns)")
        self.__features = features

    @property
    def feature_names(self):
        return self.__feature_names_in_

    @property
    def target(self):
        return self.__target

    @property
    def pipeline(self):
        return self.__pipeline

    @property
    def label_encoder(self):
        return self.__label_encoder
