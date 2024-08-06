from typing import Optional, Union, List, Tuple, Literal

from functools import lru_cache
from itertools import chain
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

FILTERS = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'


@lru_cache(maxsize=None)
def text_to_word_sequence(text, filters=FILTERS, lower=True, split=" "):
    text = text.lower() if lower else text
    filters = filters if not filters else filters
    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    seq = text.split(split)
    return [i for i in seq if i]


def count_words(
    tokens: List[tuple[str]],
    min_count: int = None,
    filters=FILTERS,
    return_dataframe: bool = False,
) -> dict:
    """
    Count the occurrences of words in a list of tokenized texts or strings.

    Args:
        data : List of tokenized texts (list of tuples or list of strings).
        min_count : Minimum count threshold for words to be included in the result, by default None.
        return_dataframe : Whether to return the result as a pandas DataFrame, by default False.

    Returns:
        Union[dict, pd.DataFrame]:
            If return_dataframe is False, returns a dictionary where keys are words and values are counts.
            If return_dataframe is True, returns a DataFrame with columns "words" and "counts".
    """
    assert any(
        isinstance(x, (str, list, tuple)) for x in tokens
    ), f"tokens is invalid! {type(tokens)}"
    if all(isinstance(x, str) for x in tokens):
        tokens = [text_to_word_sequence(text, filters=filters) for text in tokens]
    counter_words = Counter(list(chain(*tokens)))
    if min_count:
        counter_words = {
            word: count for word, count in counter_words.items() if count >= min_count
        }
        counter_words = dict(
            sorted(counter_words.items(), key=lambda item: item[1], reverse=True)
        )

    if return_dataframe:
        import pandas as pd

        return pd.DataFrame(
            {
                "words": list(counter_words.keys()),
                "counts": list(counter_words.values()),
            }
        ).sort_values(by=["counts"], ascending=False, ignore_index=True)
    return counter_words


class Tokenizer:
    """
    A class to tokenize text data, transform it into sequences, and pad sequences.

    Attributes:
        min_count : Minimum frequency count for words to be included in the vocabulary.
        filters : Characters to filter out from the text.
        oov_token : Token to use for out-of-vocabulary words.
        maxlen : Maximum length of sequences. If 0, it will be calculated based on the data.
        padding : Padding type ("pre" or "post").
        truncating : Truncating type ("pre" or "post").
        value : Value used for padding.
        dtype : Data type of the padded sequences.

    Methods:
        fit(data: Union[List[str], List[Tuple[str]]], y=None)
            Fits the tokenizer on the given data.
        transform(data)
            Transforms the given data into padded sequences.
        encode(text: str) -> list
            Encodes a single text string into a sequence of integers.
        decode(token: list) -> str
            Decodes a sequence of integers back into a text string.
        texts_to_sequences(text: list) -> list
            Converts a list of text strings into sequences of integers.
        sequences_to_texts(sequences: list) -> list
            Converts a list of sequences of integers back into text strings.
        sequences_to_pad(sequences: list)
            Pads a list of sequences to the maximum length.

    Properties:
        word_index : A dictionary mapping words to their integer indices.
        index_word : A dictionary mapping integer indices to their corresponding words.
    """

    def __init__(
        self,
        oov_token: Optional[str] = None,
        filters: Optional[str] = '!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower: bool = True,
        split=" ",
        min_count: int = 0,
        maxlen: int = 0,
        padding: Literal["pre", "post"] = "pre",
        truncating: Literal["pre", "post"] = "pre",
        value: int = 0,
        dtype="int32",
    ):
        """
        Initializes the Tokenizer with the specified parameters.

        Args:
            oov_token : Token to use for out-of-vocabulary words (default is None).
            filters : Characters to filter out from the text (default is string.punctuation).
            min_count : Minimum frequency count for words to be included in the vocabulary (default is None).
            maxlen : Maximum length of sequences (default is 0, which means it will be calculated based on the data).
            padding : Padding type ("pre" or "post", default is "pre").
            truncating : Truncating type ("pre" or "post", default is "pre").
            value : Value used for padding (default is 0).
            lower : Whether the text will be changed to lowercase.
            split : Character to be used for token splitting.
            dtype : Data type of the padded sequences (default is "int32").
        """
        assert padding in ["pre", "post"] and truncating in [
            "pre",
            "post",
        ], f"padding/trunc not found!"
        self.min_count = min_count
        self.filters = filters
        self.lower = lower
        self.oov_token = oov_token
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating
        self.value = value
        self.dtype = dtype
        self.split = split

    def fit(self, texts: List[str], y=None):
        """
        Fits the tokenizer on the given data.

        Args:
            data : The input data to fit the tokenizer on.
            y : Not used, present for compatibility with sklearn API.

        Returns:
            Tokenizer: The fitted Tokenizer object.
        """
        f_word_tokenize = lambda x: text_to_word_sequence(
            x, filters=self.filters, split=self.split, lower=self.lower
        )
        tokens = [f_word_tokenize(text) for text in texts]
        vocabs = list(count_words(tokens, min_count=self.min_count).keys())
        if self.oov_token:
            vocabs.insert(0, self.oov_token)
        self.__word_index = {word: idx for idx, word in enumerate(vocabs, start=1)}
        self.__index_word = {idx: word for idx, word in enumerate(vocabs, start=1)}

        self.__vocabs = vocabs
        self.f_word_tokenize = f_word_tokenize

        if self.maxlen == 0:
            self.maxlen = max([len(token) for token in tokens])

        return self

    def __encode(self, tokens: list) -> list:
        if self.oov_token:
            return [self.__word_index.get(token, 1) for token in tokens]
        else:
            return [
                self.__word_index[token] for token in tokens if token in self.__vocabs
            ]

    def __decode(self, token: List[int]) -> str:
        return " ".join(
            [self.__index_word[idx] for idx in token if idx in self.__index_word]
        )

    def transform(self, data):
        """
        Transforms the given data into padded sequences.

        Args:
            data : The input data to transform.

        Returns:
            np.ndarray: The transformed and padded sequences.
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.values.reshape(-1)
        return self.texts_to_sequences(data)

    def texts_to_sequences(self, texts: list) -> list:
        """
        Converts a list of text strings into sequences of integers.

        Args:
            text : The list of text strings to convert.

        Returns:
            list: The list of sequences of integers.
        """
        tokens = [self.f_word_tokenize(token) for token in texts]
        return [self.__encode(t) for t in tokens]

    def sequences_to_texts(self, sequences: List[int]) -> List[str]:
        """
        Converts a list of sequences of integers back into text strings.

        Args:
            sequences : The list of sequences of integers to convert.

        Returns:
            list: The list of decoded text strings.
        """
        return [self.__decode(s) for s in sequences]

    def sequences_to_pad(self, sequences: list):
        """
        Pads a list of sequences to the maximum length.

        Args:
            sequences : The list of sequences to pad.

        Returns:
            np.ndarray: The padded sequences.
        """
        padded_sequences = np.full(
            (len(sequences), self.maxlen), self.value, dtype=self.dtype
        )

        for i, seq in enumerate(sequences):
            if self.truncating == "pre":
                trunc = seq[-self.maxlen :]
            elif self.truncating == "post":
                trunc = seq[: self.maxlen]
            else:
                raise ValueError(f'Truncating type "{self.truncating}" not understood')

            if self.padding == "pre":
                padded_sequences[i, -len(trunc) :] = trunc
            elif self.padding == "post":
                padded_sequences[i, : len(trunc)] = trunc
            else:
                raise ValueError(f'Padding type "{self.padding}" not understood')

        return padded_sequences

    @property
    def word_index(self):
        return self.__word_index

    @property
    def index_word(self):
        return self.__index_word

    @property
    def vocab(self):
        return self.__vocabs

    @property
    def vocab_size(self):
        return len(self.__vocabs)
