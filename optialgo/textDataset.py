from . import (
    Dataset,
    check_missing_value,
    text_clean as f_text_clean,
    text_manipulation as f_text_manipulation,
    check_imbalance,
    Tokenizer,
)
import pandas as pd
from typing import Union, Optional, Literal, Callable
import numpy as np


class TextDataset:
    """
    A class to handle text data preprocessing and manipulation for machine learning tasks.

    """

    def __str__(self):
        return "textDataset"

    def __init__(self, dataframe: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        """
        Initializes the TextDataset with a dataframe, test size, and random seed.

        Args:
            dataframe: The input dataframe containing the dataset.
            test_size: Proportion of the dataset to include in the test split (default is 0.2).
            seed: Random seed for reproducibility (default is 42).

        Raises:
            ValueError: If there are missing values in the dataframe.
        """
        self.test_size = test_size
        self.seed = seed

        mv = check_missing_value(dataframe=dataframe)
        if mv:
            raise ValueError(mv)
        self.__dataframe = dataframe

    def flow_from_dataframe(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transforms the feature column of the given dataframe using the preprocessing pipeline.

        Args:
            X : The input dataframe to transform.

        Returns:
            The transformed feature column.

        Raises:
            KeyError : If the feature column is not in the dataframe.
        """
        if self.feature not in X.columns:
            raise KeyError("feature not in X")
        return self.pipeline.transform(X[self.feature])

    def flow_from_array(self, X: np.ndarray) -> np.ndarray:
        """
        Transforms the given array using the preprocessing pipeline.

        Args:
            X : The input array to transform.

        Returns:
            The transformed array.
        """
        if X.ndim == 2:
            X = X.reshape(-1)
        return self.pipeline.transform(X).toarray()

    def get_label(self, y_pred: np.ndarray) -> np.ndarray:
        """
        Converts predicted labels back to their original form using the label encoder.

        Args:
            y_pred : The predicted labels.

        Returns:
            The original labels.

        Raises:
            ValueError: If the label encoder is not found.
        """
        if not hasattr(self, "label_encoder"):
            raise ValueError("label_encoder not found!")
        if y_pred.dtype != int:
            y_pred = y_pred.astype("int32")
        return self.label_encoder.inverse_transform(y_pred)

    def __preprocessing(dataframe: pd.DataFrame, vectorizer):
        """
        Preprocesses the dataframe using the specified vectorizer.

        Args:
            dataframe : The input dataframe to preprocess.
            vectorizer : The vectorizer to use for preprocessing.

        Returns:
            tuple
                The preprocessing pipeline and vectorizer.

        Raises:
            TypeError: If the vectorizer is not found.
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.pipeline import Pipeline

        if isinstance(vectorizer, str):
            if vectorizer == "tfidf":
                vectorizer = TfidfVectorizer()
            elif vectorizer == "count_vect":
                vectorizer = CountVectorizer()
            else:
                raise TypeError("vectorizer not found !")
        steps = [("vectorizer", vectorizer)]

        pipeline = Pipeline(steps=steps)
        pipeline.fit(dataframe)
        return pipeline, vectorizer

    def get_x_y(self) -> tuple:
        """
        Returns the train and test splits of features and labels.

        Returns:
            The training features, testing features, training labels, and testing labels.
        """
        X_train = self.train[:, :-1]
        y_train = self.train[:, -1]
        X_test = self.test[:, :-1]
        y_test = self.test[:, -1]
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        feature: str,
        target: Optional[str],
        t: Literal["classification", "regression"],
        vectorizer: Union[Literal["tfidf", "count_vect"], Tokenizer] = "tfidf",
        ci: bool = False,
    ):
        """
        Fits the preprocessing pipeline and prepares the data for training and testing.

        Args:
            feature: The feature column to use for training.
            target : The target column to use for training.
            t : The type of task (classification or regression).
            vectorizer : The vectorizer to use for preprocessing (default is "tfidf").
            verbose : Whether to print verbose output (default is False).
            ci : Whether to check for class imbalance (default is False).

        Returns:
            TextDataset: The fitted TextDataset object.
        """
        from sklearn.preprocessing import LabelEncoder

        dataframe = self.dataframe.copy()

        stratify = None

        if target:
            if t == "classification":
                if ci:
                    check_imbalance(dataframe=dataframe, target=target)
                label_encoder = LabelEncoder().fit(dataframe[target].values)
                dataframe[target] = label_encoder.transform(dataframe[target].values)
                self.__label_encoder = label_encoder
                self.class_type = (
                    "binary"
                    if len(dataframe[target].value_counts().values) == 2
                    else "multiclass"
                )
        self.__target = target

        self.t = t

        train, test = Dataset.train_test_split(
            dataframe, test_size=self.test_size, seed=self.seed, stratify=stratify
        )

        pipeline, vectorizer = TextDataset.__preprocessing(train[feature], vectorizer)
        self.__feature = feature
        self.__pipeline = pipeline
        self.__vectorizer = vectorizer
        _train = (
            pipeline.transform(train[feature])
            if isinstance(vectorizer, Tokenizer)
            else pipeline.transform(train[feature]).toarray()
        )
        _test = (
            pipeline.transform(test[feature])
            if isinstance(vectorizer, Tokenizer)
            else pipeline.transform(test[feature]).toarray()
        )

        if target:
            _train = np.hstack((_train, train[[target]].values))
            _test = np.hstack((_test, test[[target]].values))

        self.__train = _train
        self.__test = _test
        self.feature_names = [feature]
        return self

    @property
    def feature(self):
        return self.__feature

    @property
    def target(self):
        return self.__target

    @property
    def dataframe(self):
        return self.__dataframe

    @property
    def train(self):
        return self.__train

    @property
    def test(self):
        return self.__test

    @property
    def pipeline(self):
        return self.__pipeline

    @property
    def vectorizer(self):
        return self.__vectorizer

    @property
    def label_encoder(self):
        return self.__label_encoder
