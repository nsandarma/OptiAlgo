from . import (
    Dataset,
    check_missing_value,
    text_clean,
    text_manipulation,
    check_imbalance,
    Tokenizer,
)
import pandas as pd
from typing import Union, Optional, Literal, List
import numpy as np


class TextDataset:
    def __str__(self):
        return "textDataset"

    def __init__(self, dataframe: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
        self.test_size = test_size
        self.seed = seed

        mv = check_missing_value(dataframe=dataframe)
        if mv:
            raise ValueError(mv)
        self.__dataframe = dataframe

    def flow_from_dataframe(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature not in X.columns:
            raise KeyError("feature not in X")
        return self.pipeline.transform(X[self.feature])

    def flow_from_array(self, X: np.ndarray):
        # X = pd.DataFrame(X, columns=[self.feature])
        if X.ndim == 2:
            X = X.reshape(-1)
        return self.pipeline.transform(X).toarray()

    def get_label(self, y_pred: np.ndarray):
        if not hasattr(self, "label_encoder"):
            raise ValueError("label_encoder not found!)")
        if y_pred.dtype != int:
            y_pred = y_pred.astype("int32")
        return self.label_encoder.inverse_transform(y_pred)

    def __preprocessing(dataframe: pd.DataFrame, vectorizer):
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
        X_train = self.train[:, :-1]
        y_train = self.train[:, -1]
        X_test = self.test[:, :-1]
        y_test = self.test[:, -1]
        return X_train, X_test, y_train, y_test

    def fit(
        self,
        feature: str,
        target: str,
        lang: str,
        t: Literal["classification", "regression", "clustering"],
        vectorizer: Union[Literal["tfidf", "count_vect"], Tokenizer] = "tfidf",
        verbose: bool = False,
        ci: bool = False,
    ):

        from sklearn.preprocessing import LabelEncoder

        dataframe = self.dataframe.copy()

        texts = text_clean(
            dataframe[feature].tolist(), return_token=True, verbose=verbose
        )
        texts = text_manipulation(texts, lang=lang, verbose=verbose)
        dataframe[feature] = texts

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
