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

    Attributes
    ----------
    test_size : float
        Proportion of the dataset to include in the test split.
    seed : int
        Random seed for reproducibility.
    feature : str
        The feature column used for model training.
    target : str
        The target column used for model training.
    dataframe : pd.DataFrame
        The original dataframe containing the dataset.
    train : np.ndarray
        The training data.
    test : np.ndarray
        The testing data.
    pipeline : sklearn.pipeline.Pipeline
        The preprocessing pipeline.
    vectorizer : sklearn.feature_extraction.text.Vectorizer
        The vectorizer used in the pipeline.
    label_encoder : sklearn.preprocessing.LabelEncoder
        The label encoder used for target labels.

    Methods
    -------
    flow_from_dataframe(X: pd.DataFrame) -> np.ndarray
        Transforms the feature column of the given dataframe using the preprocessing pipeline.
    flow_from_array(X: np.ndarray)
        Transforms the given array using the preprocessing pipeline.
    get_label(y_pred: np.ndarray) -> np.ndarray
        Converts predicted labels back to their original form using the label encoder.
    fit(feature: str, target: Optional[str], lang: str, t: Literal["classification", "regression"], vectorizer: Union[Literal["tfidf", "count_vect"], Tokenizer] = "tfidf", verbose: bool = False, ci: bool = False)
        Fits the preprocessing pipeline and prepares the data for training and testing.
    get_x_y() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Returns the train and test splits of features and labels.
    """

    def __str__(self):
        return "textDataset"

    def __init__(self, dataframe: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        """
        Initializes the TextDataset with a dataframe, test size, and random seed.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe containing the dataset.
        test_size : float, optional
            Proportion of the dataset to include in the test split (default is 0.2).
        seed : int, optional
            Random seed for reproducibility (default is 42).

        Raises
        ------
        ValueError
            If there are missing values in the dataframe.
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

        Parameters
        ----------
        X : pd.DataFrame
            The input dataframe to transform.

        Returns
        -------
        np.ndarray
            The transformed feature column.

        Raises
        ------
        KeyError
            If the feature column is not in the dataframe.
        """
        if self.feature not in X.columns:
            raise KeyError("feature not in X")
        return self.pipeline.transform(X[self.feature])

    def flow_from_array(self, X: np.ndarray):
        """
        Transforms the given array using the preprocessing pipeline.

        Parameters
        ----------
        X : np.ndarray
            The input array to transform.

        Returns
        -------
        np.ndarray
            The transformed array.
        """
        if X.ndim == 2:
            X = X.reshape(-1)
        return self.pipeline.transform(X).toarray()

    def get_label(self, y_pred: np.ndarray):
        """
        Converts predicted labels back to their original form using the label encoder.

        Parameters
        ----------
        y_pred : np.ndarray
            The predicted labels.

        Returns
        -------
        np.ndarray
            The original labels.

        Raises
        ------
        ValueError
            If the label encoder is not found.
        """
        if not hasattr(self, "label_encoder"):
            raise ValueError("label_encoder not found!")
        if y_pred.dtype != int:
            y_pred = y_pred.astype("int32")
        return self.label_encoder.inverse_transform(y_pred)

    def __preprocessing(dataframe: pd.DataFrame, vectorizer):
        """
        Preprocesses the dataframe using the specified vectorizer.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The input dataframe to preprocess.
        vectorizer : Union[str, TfidfVectorizer, CountVectorizer]
            The vectorizer to use for preprocessing.

        Returns
        -------
        tuple
            The preprocessing pipeline and vectorizer.

        Raises
        ------
        TypeError
            If the vectorizer is not found.
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

    def get_x_y(self):
        """
        Returns the train and test splits of features and labels.

        Returns
        -------
        tuple
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
        ci: bool = False
    ):
        """
        Fits the preprocessing pipeline and prepares the data for training and testing.

        Parameters
        ----------
        feature : str
            The feature column to use for training.
        target : Optional[str]
            The target column to use for training.
        t : Literal["classification", "regression"]
            The type of task (classification or regression).
        vectorizer : Union[Literal["tfidf", "count_vect"], Tokenizer], optional
            The vectorizer to use for preprocessing (default is "tfidf").
        verbose : bool, optional
            Whether to print verbose output (default is False).
        ci : bool, optional
            Whether to check for class imbalance (default is False).

        Returns
        -------
        TextDataset
            The fitted TextDataset object.
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
        """
        Returns the feature column name.

        Returns
        -------
        str
            The feature column name.
        """
        return self.__feature

    @property
    def target(self):
        """
        Returns the target column name.

        Returns
        -------
        str
            The target column name.
        """
        return self.__target

    @property
    def dataframe(self):
        """
        Returns the original dataframe.

        Returns
        -------
        pd.DataFrame
            The original dataframe.
        """
        return self.__dataframe

    @property
    def train(self):
        """
        Returns the training data.

        Returns
        -------
        np.ndarray
            The training data.
        """
        return self.__train

    @property
    def test(self):
        """
        Returns the testing data.

        Returns
        -------
        np.ndarray
            The testing data.
        """
        return self.__test

    @property
    def pipeline(self):
        """
        Returns the preprocessing pipeline.

        Returns
        -------
        sklearn.pipeline.Pipeline
            The preprocessing pipeline.
        """
        return self.__pipeline

    @property
    def vectorizer(self):
        """
        Returns the vectorizer used in the pipeline.

        Returns
        -------
        sklearn.feature_extraction.text.Vectorizer
            The vectorizer used in the pipeline.
        """
        return self.__vectorizer

    @property
    def label_encoder(self):
        """
        Returns the label encoder used for target labels.

        Returns
        -------
        sklearn.preprocessing.LabelEncoder
            The label encoder used for target labels.

        Raises
        ------
        ValueError
            If the label encoder is not found.
        """
        return self.__label_encoder
