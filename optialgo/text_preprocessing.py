import string
import re
from typing import Union, List, Optional, Pattern, Set, Tuple, Literal
from nltk.corpus import stopwords as st
from functools import lru_cache


@lru_cache(maxsize=None)
def f_remove_punctuation(text: str, punctuation: str = string.punctuation) -> str:
    """
    Remove punctuation from a single text string.

    Parameters
    ----------
    text : str
        The input text from which to remove punctuation.
    punctuation : str, optional
        The punctuation characters to remove, by default string.punctuation.

    Returns
    -------
    str
        The text with punctuation removed.
    """
    translation = str.maketrans(punctuation, " " * len(punctuation))
    return text.translate(translation)


def remove_punctuation(
    data: List[str], punctuation: str = string.punctuation
) -> List[str]:
    """
    Remove punctuation from a list of texts.

    Parameters
    ----------
    data : List[str]
        List of texts.
    punctuation : str, optional
        The punctuation characters to remove, by default string.punctuation.

    Returns
    -------
    List[str]
        List of texts without punctuation.
    """
    return [f_remove_punctuation(text, punctuation) for text in data]


def f_remove_digits(text: str) -> str:
    """
    Remove digits from a single text string.

    Parameters
    ----------
    text : str
        The input text from which to remove digits.

    Returns
    -------
    str
        The text with digits removed.
    """
    return re.sub(r"\d+", "", text)


def remove_digits(data: List[str]) -> List[str]:
    """
    Remove digits from a list of texts.

    Parameters
    ----------
    data : List[str]
        List of texts.

    Returns
    -------
    List[str]
        List of texts without digits.
    """
    return [f_remove_digits(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_url(text) -> str:
    """
    Remove URLs from a single text string.

    Parameters
    ----------
    text : str
        The input text from which to remove URLs.

    Returns
    -------
    str
        The text with URLs removed.
    """
    return re.compile(r"https?://\S+|www\.\S+").sub(r"", text)


def remove_url(data: List[str]) -> List[str]:
    """
    Remove URLs from a list of texts.

    Parameters
    ----------
    data : List[str]
        List of texts.

    Returns
    -------
    List[str]
        List of texts without URLs.
    """
    return [f_remove_url(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_emoji(text: str) -> str:
    """
    Remove emojis from a single text string.

    Parameters
    ----------
    text : str
        The input text from which to remove emojis.

    Returns
    -------
    str
        The text with emojis removed.
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F700-\U0001F77F"  # alchemical symbols
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"", text)


def remove_emoji(data: List[str]) -> List[str]:
    """
    Remove emojis from a list of texts.

    Parameters
    ----------
    data : List[str]
        List of texts.

    Returns
    -------
    List[str]
        List of texts without emojis.
    """
    return [f_remove_emoji(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_non_latin(text: str) -> str:
    """
    Remove non-Latin characters from a single text string.

    Parameters
    ----------
    text : str
        The input text from which to remove non-Latin characters.

    Returns
    -------
    str
        The text with non-Latin characters removed.
    """
    return re.sub(r"[^\x00-\x7F]+", "", text)


def remove_non_latin(data: List[str]) -> List[str]:
    """
    Remove non-Latin characters from a list of texts.

    Parameters
    ----------
    data : List[str]
        List of texts.

    Returns
    -------
    List[str]
        List of texts without non-Latin characters.
    """
    return [f_remove_non_latin(text) for text in data]


F_TEXT_CLEAN = [
    "remove_punctuation",
    "remove_digits",
    "remove_url",
    "remove_emoji",
    "remove_non_latin",
]


def lower_text(data: List[str]) -> List[str]:
    return [x.lower() for x in data]


def f_regex_word_tokenize(text: str, pattern: Pattern[str]) -> Tuple[str]:
    """
    Tokenize a text string using a regex pattern.

    Parameters
    ----------
    text : str
        The input text to tokenize.
    pattern : Pattern[str]
        The regex pattern to use for tokenization.

    Returns
    -------
    Tuple[str]
        A tuple of tokens.
    """
    from nltk.tokenize import regexp_tokenize

    return tuple(regexp_tokenize(text=text, pattern=pattern))


def f_word_tokenize(text: str):
    """
    Tokenize a text string using NLTK's word_tokenize.

    Parameters
    ----------
    text : str
        The input text to tokenize.

    Returns
    -------
    Tuple[str]
        A tuple of tokens.
    """
    from nltk.tokenize import word_tokenize as wt

    return tuple(wt(text))


def word_tokenize(
    data: List[str], pattern: Optional[Pattern[str]] = None
) -> List[Tuple[str, ...]]:
    """
    Tokenize a list of text strings.

    Parameters
    ----------
    data : List[str]
        List of texts to tokenize.
    pattern : Optional[Pattern[str]], optional
        The regex pattern to use for tokenization, by default None.

    Returns
    -------
    List[Tuple[str, ...]]
        List of tuples of tokens.
    """
    if pattern:
        return [f_regex_word_tokenize(x, pattern=pattern) for x in data]

    return [f_word_tokenize(x) for x in data]


def token_to_str(data: List[Tuple[str]]) -> List[str]:
    """
    Convert a list of token tuples to a list of strings.

    Parameters
    ----------
    data : List[Tuple[str]]
        List of token tuples.

    Returns
    -------
    List[str]
        List of strings joined from tokens.
    """
    return [" ".join(x) for x in data]


def find_duplicates(data: List[str]) -> dict:
    """
    Find duplicates in a list of strings and return their indices.

    Parameters
    ----------
    data : List[str]
        List of strings to check for duplicates.

    Returns
    -------
    dict
        A dictionary where keys are duplicate strings and values are lists of their indices.
    """
    from collections import defaultdict

    index_map = defaultdict(list)
    duplicates = defaultdict(list)

    for index, item in enumerate(data):
        index_map[item].append(index)
        if len(index_map[item]) > 1:
            duplicates[item] = index_map[item]

    return dict(duplicates)


def text_clean(
    data: List[str],
    punctuation: Optional[str] = string.punctuation,
    lower: bool = True,
    digits: bool = True,
    emoji: bool = True,
    duplicates: bool = True,
    url: bool = True,
    non_latin: bool = True,
    return_token: bool = False,
    return_dataframe: bool = False,
    verbose: bool = False,
    pattern: Optional[Pattern[str]] = None,
):
    """
    Clean a list of text data based on specified parameters.

    Parameters
    ----------
    data : List[str]
        List of strings to clean.

    punctuation : str, optional
        Punctuation characters to remove, by default string.punctuation.

    lower : bool, optional
        Convert text to lowercase, by default True.

    digits : bool, optional
        Remove digits from text, by default True.

    emoji : bool, optional
        Remove emojis from text, by default True.

    duplicates : bool, optional
        Remove duplicate entries from data, by default True.

    url : bool, optional
        Remove URLs from text, by default True.

    non_latin : bool, optional
        Remove non-Latin characters from text, by default True.

    return_token : bool, optional
        Tokenize text and return tokens, by default False.

    return_dataframe : bool, optional
        Return result as a Pandas DataFrame, by default False.

    verbose : bool, optional
        Display progress using tqdm, by default False.

    pattern : Optional[Pattern[str]], optional
        Regex pattern for tokenization, by default None.

    Returns
    -------
    List[str]
        Cleaned list of text data.

    Raises
    ------
    ValueError
        If return_dataframe is True but lengths of data and cleaned data don't match.
    """
    from tqdm import tqdm

    def maybe_tqdm(iterable, desc):
        return tqdm(iterable, desc=desc) if verbose else iterable

    idx = None
    if duplicates:
        idx = find_duplicates(data)
        datac = list(dict.fromkeys(data))
        data = datac.copy()
    else:
        datac = data[:]
    if lower:
        datac = [x.lower() for x in maybe_tqdm(datac, "Case Folding")]

    if emoji:
        datac = remove_emoji(maybe_tqdm(datac, "Removing Emoji"))

    if url:
        datac = remove_url(maybe_tqdm(datac, "Removing URLs"))

    if punctuation:
        datac = remove_punctuation(
            maybe_tqdm(datac, "Removing Punctuation"), punctuation=punctuation
        )

    if digits:
        datac = remove_digits(maybe_tqdm(datac, "Removing Digits"))

    if non_latin:
        datac = remove_non_latin(maybe_tqdm(datac, "Removing Non-Latin Characters"))

    if return_token:
        datac = word_tokenize(maybe_tqdm(datac, "Tokenizing"), pattern=pattern)

    if return_dataframe:
        import pandas as pd

        if len(data) == len(datac):
            return pd.DataFrame({"raw": data, "pre": datac})
        print("there is duplicate data")
        print(idx.values())

    return datac


def get_stopwords_en() -> List[str]:
    """
    Get English stopwords from NLTK corpus.

    Returns
    -------
    List[str]
        List of English stopwords.
    """
    return st.words("english")


def get_stopwords_idn() -> List[str]:
    """
    Get Indonesian stopwords from NLTK corpus.

    Returns
    -------
    List[str]
        List of Indonesian stopwords.
    """
    return st.words("indonesian")


def f_stopwords(
    token: Union[Tuple[str], str], stopwords: Set[str], return_token: bool = False
) -> Union[Tuple[str, ...], str]:
    """
    Remove stopwords from a token or tuple of tokens.

    Parameters
    ----------
    token : Union[Tuple[str], str]
        Token or tuple of tokens to filter.
    stopwords : Set[str]
        Set of stopwords to filter out.
    return_token : bool, optional
        Whether to return tokens (True) or a joined string (False), by default False.

    Returns
    -------
    Union[Tuple[str, ...], str]
        Filtered tokens or joined string without stopwords.
    """
    if isinstance(token, str):
        token = f_word_tokenize(token)
    filtered_words = tuple([word for word in token if word not in stopwords])
    if return_token:
        return filtered_words
    return " ".join(filtered_words)


def remove_stopwords(
    tokens: List[Tuple[str]],
    lang: str,
    stopwords: Optional[List[str]] = None,
    additional: Optional[List[str]] = None,
    return_token: bool = True,
    verbose: bool = False,
) -> Union[List[str], List[Tuple[str]]]:
    """
    Remove stopwords from a list of tokenized texts.

    Parameters
    ----------
    tokens : List[Union[Tuple[str], List[str]]]
        List of tokenized texts, where each item can be a tuple or list of tokens.
    lang : str
        Language code for stopwords ('english' or 'indonesian').
    stopwords : Optional[List[str]], optional
        List of additional stopwords to remove, by default None.
    additional : Optional[List[str]], optional
        List of additional stopwords to add, by default None.
    return_token : bool, optional
        Whether to return tokens (True) or joined strings (False), by default True.
    verbose : bool, optional
        Whether to display progress bar, by default False.

    Returns
    -------
    Union[List[str], List[Tuple[str]]]
        List of tokenized texts with stopwords removed.
    """
    from tqdm import tqdm

    if not any(isinstance(x, (list, tuple)) for x in tokens):
        raise ValueError("text data must be tokenized first")
    if not stopwords:
        if lang == "indonesian":
            stopwords = get_stopwords_idn()
        elif lang == "english":
            stopwords = get_stopwords_en()
        else:
            raise ValueError("lang not found !")
        if additional:
            stopwords += additional

    return [
        f_stopwords(token=x, stopwords=set(stopwords), return_token=return_token)
        for x in tqdm(tokens, desc="Stopwords", disable=not verbose)
    ]


def get_norm_words_idn():
    import os
    import pickle

    file_path = os.path.join(os.path.dirname(__file__), "nlp_equipments", "kbba.csv")
    with open(file_path, "rb") as file:
        norm_words = pickle.load(file)

    return norm_words


@lru_cache(maxsize=None)
def f_normalize(
    token: Union[Tuple[str], str], norm_words: dict, return_token: bool = False
) -> Union[Tuple[str, ...], str]:
    """
    Normalize tokens using a dictionary of normalization mappings.

    Parameters
    ----------
    token : Union[Tuple[str], str]
        Token or tuple of tokens to normalize.
    norm_words : dict
        Dictionary mapping tokens to their normalized forms.
    return_token : bool, optional
        Whether to return tokens (True) or a joined string (False), by default False.

    Returns
    -------
    Union[Tuple[str, ...], str]
        Normalized tokens or joined string of normalized tokens.
    """
    if isinstance(token, str):
        token = f_word_tokenize(token)

    result = tuple(norm_words.get(x, x) for x in token)
    if return_token:
        return result
    return " ".join(result)


def normalize(
    tokens: List[Tuple[str]],
    norm_words: dict,
    return_token: bool = True,
    verbose: bool = False,
):
    """
    Normalize a list of tokenized texts using a dictionary of normalization mappings.

    Parameters
    ----------
    tokens : List[Union[Tuple[str], List[str]]]
        List of tokenized texts, where each item can be a tuple or list of tokens.
    norm_words : dict
        Dictionary mapping tokens to their normalized forms.
    return_token : bool, optional
        Whether to return tokens (True) or joined strings (False), by default True.
    verbose : bool, optional
        Whether to display progress bar, by default False.

    Returns
    -------
    List[Union[str, Tuple[str]]]
        List of normalized texts.
    """
    from tqdm import tqdm

    if not any(isinstance(x, (list, tuple)) for x in tokens):
        raise ValueError("text data must be tokenized first")

    return [
        f_normalize(token=x, norm_words=norm_words, return_token=return_token)
        for x in tqdm(tokens, desc="normalize tokens", disable=not verbose)
    ]


def __get_lemmatizer(lang: str):
    from nlp_id.lemmatizer import Lemmatizer
    from nltk.stem import WordNetLemmatizer

    if lang == "indonesian":
        return Lemmatizer()
    return WordNetLemmatizer()


def f_lemmatization_idn(text: str, return_token: bool = False) -> Union[Tuple]:
    """
    Lemmatize Indonesian text.

    Parameters
    ----------
    text : str
        Input text to lemmatize.
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.

    Returns
    -------
    Union[Tuple[str], str]
        If return_token is True, returns a tuple of lemmatized tokens.
        If return_token is False, returns the lemmatized text as a string.
    """
    text = __get_lemmatizer(lang="indonesian").lemmatize(text)
    if return_token:
        return f_word_tokenize(text)
    return text


def __get_wordnet_pos(tag: str) -> str:
    from nltk.corpus import wordnet

    tag = tag[0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def f_lemmatization_en(
    token: Union[Tuple[str], str], return_token: bool = False
) -> Union[Tuple[str, ...], str]:
    """
    Lemmatize English tokens based on their part-of-speech tags using a lemmatizer.

    Parameters
    ----------
    token : Union[Tuple[str], str]
        Token or tuple of tokens to lemmatize.
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.

    Returns
    -------
    Union[Tuple[str, ...], str]
        If return_token is True, returns a tuple of lemmatized tokens.
        If return_token is False, returns the lemmatized text as a joined string.
    """
    import nltk

    if isinstance(token, str):
        token = f_word_tokenize(token)
    pos_tags = nltk.pos_tag(token)
    lemmatized_words = tuple(
        __get_lemmatizer(lang="english").lemmatize(word, __get_wordnet_pos(tag))
        for word, tag in pos_tags
    )
    if not return_token:
        return " ".join(lemmatized_words)
    return lemmatized_words


def __get_stemmer(lang: str):
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from nltk.stem import PorterStemmer

    if lang == "indonesian":
        return StemmerFactory().create_stemmer()
    return PorterStemmer()


def f_stemming_idn(
    text: str, return_token: bool = False
) -> Union[Tuple[str, ...], str]:
    """
    Perform stemming on Indonesian text.

    Parameters
    ----------
    text : str
        Input text to stem.
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.

    Returns
    -------
    Union[Tuple[str, ...], str]
        If return_token is True, returns a tuple of stemmed tokens.
        If return_token is False, returns the stemmed text as a joined string.
    """
    text = __get_stemmer(lang="indonesian").stem(text)
    if return_token:
        return f_word_tokenize(text)
    return text


def f_stemming_en(text: str, return_token: bool = False) -> Union[Tuple[str, ...], str]:
    """
    Perform stemming on English text.

    Parameters
    ----------
    text : str
        Input text to stem.
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.

    Returns
    -------
    Union[Tuple[str, ...], str]
        If return_token is True, returns a tuple of stemmed tokens.
        If return_token is False, returns the stemmed text as a joined string.
    """
    if isinstance(text, str):
        text = f_word_tokenize(text)
    stemmed_words = tuple(__get_stemmer(lang="english").stem(word) for word in text)
    if return_token:
        return stemmed_words
    return " ".join(stemmed_words)


def lemmatization(
    data: Union[List[Tuple[str]], List[str]],
    lang: str,
    return_token: bool = False,
    verbose: bool = False,
):
    """
    Lemmatize tokenized texts based on language.

    Parameters
    ----------
    data : Union[List[Tuple[str]], List[str]]
        List of tokenized texts, where each item can be a tuple or list of tokens.
    lang : str
        Language code for lemmatization ('indonesian' or 'english').
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.
    verbose : bool, optional
        Whether to display progress bar, by default False.

    Returns
    -------
    List[Union[Tuple[str], str]]
        List of lemmatized texts.
        If return_token is True, each item is a tuple of lemmatized tokens.
        If return_token is False, each item is a joined string of lemmatized tokens.
    """
    if lang not in ("indonesian", "english"):
        raise ValueError("lang not found !")
    from tqdm import tqdm

    if not any(isinstance(x, (tuple, list)) for x in data):
        raise ValueError("data is not tokenize")

    if lang == "english":
        return [
            f_lemmatization_en(token, return_token=return_token)
            for token in tqdm(data, desc="Lemmatizing English", disable=not verbose)
        ]
    else:
        lemmatizer = __get_lemmatizer(lang=lang)

        @lru_cache(maxsize=None)
        def __lemma_word(word):
            return lemmatizer.lemmatize(word)

        def __lemma_sentence(sentence):
            lemmatized = [__lemma_word(word) for word in sentence]
            if return_token:
                return lemmatized
            return " ".join(lemmatized)

        return [
            __lemma_sentence(sentence)
            for sentence in tqdm(
                data, desc="Lemmatizing Indonesian", disable=not verbose
            )
        ]


def stemming(
    data: List[Tuple[str]], lang: str, return_token: bool = False, verbose: bool = False
):
    """
    Perform stemming on tokenized texts based on language.

    Parameters
    ----------
    data : List[Tuple[str]]
        List of tokenized texts, where each item is a tuple of tokens.
    lang : str
        Language code for stemming ('indonesian' or 'english').
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.
    verbose : bool, optional
        Whether to display a progress bar, by default False.

    Returns
    -------
    List[str]
        List of stemmed texts.
        If return_token is True, each item is a list of stemmed tokens.
        If return_token is False, each item is a joined string of stemmed tokens.
    """
    from tqdm import tqdm

    if lang not in ("indonesian", "english"):
        raise ValueError("Unsupported language!")

    if not all(isinstance(x, (tuple, list)) for x in data):
        raise ValueError("Data is not tokenized!")

    stemmer = __get_stemmer(lang=lang)

    @lru_cache(maxsize=None)
    def __stem_word(word):
        return stemmer.stem(word)

    def __stem_sentence(sentence):
        stemmed_sentence = [__stem_word(word) for word in sentence]
        if return_token:
            return stemmed_sentence
        return " ".join(stemmed_sentence)

    if verbose:
        desc = f"Stemming {lang.title()}"
        data_iter = tqdm(data, desc=desc)
    else:
        data_iter = data

    # Process each sentence or tuple
    result = [__stem_sentence(sentence) for sentence in data_iter]

    return result


def text_manipulation(
    tokens: List[Tuple[str]],
    lang: str,
    stopwords: Union[List[str], bool] = False,
    stem: bool = False,
    return_dataframe: bool = False,
    norm_words: Optional[dict] = None,
    return_token=False,
    additional: Optional[List[str]] = None,
    verbose: bool = False,
):
    """
    Perform text manipulation including normalization, stopword removal, stemming or lemmatization.

    Parameters
    ----------
    tokens : List[Tuple[str]]
        List of tokenized texts, where each item is a tuple of tokens.
    lang : str
        Language code for text manipulation ('indonesian' or 'english').
    stopwords : Union[List[str], bool], optional
        List of stopwords or True to use default stopwords for the specified language, by default False.
    stem : bool, optional
        Whether to perform stemming (True) or lemmatization (False), by default False.
    return_dataframe : bool, optional
        Whether to return results as a DataFrame, by default False.
    norm_words : Optional[dict], optional
        Dictionary of normalization words, by default None.
    return_token : bool, optional
        Whether to return tokens (True) or joined string (False), by default False.
    additional : Optional[List[str]], optional
        Additional stopwords to include, by default None.
    verbose : bool, optional
        Whether to display progress bars for preprocessing steps, by default False.

    Returns
    -------
    Union[List[str], pd.DataFrame]
        If return_dataframe is False, returns a list of preprocessed texts/tokens.
        If return_dataframe is True, returns a pandas DataFrame with columns "raw" and "pre".
    """
    if not any(isinstance(x, (list, tuple)) for x in tokens):
        raise ValueError("data is not tokenizerd")

    if norm_words:
        datac = normalize(
            tokens, norm_words=norm_words, return_token=True, verbose=verbose
        )
    else:
        datac = tokens[:]

    if stopwords:
        if stopwords is True:
            datac = remove_stopwords(
                tokens,
                lang=lang,
                return_token=True,
                verbose=verbose,
                additional=additional,
            )
        else:
            datac = remove_stopwords(
                tokens,
                lang=lang,
                return_token=True,
                verbose=verbose,
                stopwords=stopwords,
            )

    if stem:
        datac = stemming(datac, lang=lang, return_token=return_token, verbose=verbose)
    else:
        datac = lemmatization(
            datac, lang=lang, return_token=return_token, verbose=verbose
        )

    if return_dataframe:
        import pandas as pd

        datac = pd.DataFrame({"raw": tokens, "pre": datac})
    return datac


class Tokenizer:
    """
    A class to tokenize text data, transform it into sequences, and pad sequences.

    Attributes
    ----------
    min_count : Optional[int]
        Minimum frequency count for words to be included in the vocabulary.
    filters : Optional[str]
        Characters to filter out from the text.
    oov_token : str
        Token to use for out-of-vocabulary words.
    maxlen : int
        Maximum length of sequences. If 0, it will be calculated based on the data.
    padding : Literal["pre", "post"]
        Padding type ("pre" or "post").
    truncating : Literal["pre", "post"]
        Truncating type ("pre" or "post").
    value : int
        Value used for padding.
    dtype : str
        Data type of the padded sequences.
    
    Methods
    -------
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
    texts_to_pad_sequences(text: list)
        Converts a list of text strings into padded sequences.
    
    Properties
    ----------
    word_index : dict
        A dictionary mapping words to their integer indices.
    index_word : dict
        A dictionary mapping integer indices to their corresponding words.
    """

    def __init__(
        self,
        oov_token: str = None,
        filters: Optional[str] = string.punctuation,
        min_count: Optional[int] = None,
        maxlen: int = 0,
        padding: Literal["pre", "post"] = "pre",
        truncating: Literal["pre", "post"] = "pre",
        value: int = 0,
        dtype="int32",
    ):
        """
        Initializes the Tokenizer with the specified parameters.

        Parameters
        ----------
        oov_token : str, optional
            Token to use for out-of-vocabulary words (default is None).
        filters : Optional[str], optional
            Characters to filter out from the text (default is string.punctuation).
        min_count : Optional[int], optional
            Minimum frequency count for words to be included in the vocabulary (default is None).
        maxlen : int, optional
            Maximum length of sequences (default is 0, which means it will be calculated based on the data).
        padding : Literal["pre", "post"], optional
            Padding type ("pre" or "post", default is "pre").
        truncating : Literal["pre", "post"], optional
            Truncating type ("pre" or "post", default is "pre").
        value : int, optional
            Value used for padding (default is 0).
        dtype : str, optional
            Data type of the padded sequences (default is "int32").
        """
        self.min_count = min_count
        self.filters = filters
        self.oov_token = oov_token
        self.maxlen = maxlen
        self.padding = padding
        self.truncating = truncating
        self.value = value
        self.dtype = dtype

    def fit(self, data: Union[List[str], List[Tuple[str]]], y=None):
        """
        Fits the tokenizer on the given data.

        Parameters
        ----------
        data : Union[List[str], List[Tuple[str]]]
            The input data to fit the tokenizer on.
        y : optional
            Not used, present for compatibility with sklearn API.

        Returns
        -------
        Tokenizer
            The fitted Tokenizer object.
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            data = data.values.reshape(-1)
        if self.filters:
            data = remove_punctuation(data, punctuation=self.filters)
        counter_words = count_words(
            data, return_dataframe=True, min_count=self.min_count
        )

        words = counter_words.words.tolist()
        self.words = words
        start = 1
        if self.oov_token:
            words.insert(0, "<OOV>")
            start = 0
        key_to_index = {word: i for i, word in enumerate(words, start=start)}
        self.__str_to_int = key_to_index
        self.__int_to_str = {i: word for word, i in key_to_index.items()}

        # Calculate maxlen if not provided
        if self.maxlen == 0:
            sequences = self.texts_to_sequences(data)
            self.maxlen = max(len(seq) for seq in sequences)

        return self

    def transform(self, data):
        """
        Transforms the given data into padded sequences.

        Parameters
        ----------
        data : Union[List[str], List[Tuple[str]]]
            The input data to transform.

        Returns
        -------
        np.ndarray
            The transformed and padded sequences.
        """
        import pandas as pd

        if isinstance(data, pd.DataFrame):
            data = data.values.reshape(-1)
        return self.texts_to_pad_sequences(data)

    def encode(self, text: str) -> list:
        """
        Encodes a single text string into a sequence of integers.

        Parameters
        ----------
        text : str
            The text string to encode.

        Returns
        -------
        list
            The encoded sequence of integers.
        """
        if self.oov_token:
            return [
                self.__str_to_int[token] if token in self.words else 0
                for token in f_word_tokenize(text)
            ]
        else:
            return [
                self.__str_to_int[token]
                for token in f_word_tokenize(text)
                if token in self.words
            ]

    def decode(self, token: list) -> str:
        """
        Decodes a sequence of integers back into a text string.

        Parameters
        ----------
        token : list
            The sequence of integers to decode.

        Returns
        -------
        str
            The decoded text string.
        """
        return " ".join([self.__int_to_str[t] for t in token if t in self.__int_to_str])

    def texts_to_sequences(self, text: list) -> list:
        """
        Converts a list of text strings into sequences of integers.

        Parameters
        ----------
        text : list
            The list of text strings to convert.

        Returns
        -------
        list
            The list of sequences of integers.
        """
        return [
            self.encode(f_remove_punctuation(s, punctuation=self.filters)) for s in text
        ]

    def sequences_to_texts(self, sequences: list) -> list:
        """
        Converts a list of sequences of integers back into text strings.

        Parameters
        ----------
        sequences : list
            The list of sequences of integers to convert.

        Returns
        -------
        list
            The list of decoded text strings.
        """
        return [self.decode(s) for s in sequences]

    def sequences_to_pad(self, sequences: list):
        """
        Pads a list of sequences to the maximum length.

        Parameters
        ----------
        sequences : list
            The list of sequences to pad.

        Returns
        -------
        np.ndarray
            The padded sequences.
        """
        import numpy as np

        maxlen = self.maxlen

        # Prepare the result array with the appropriate type and size
        padded_sequences = np.full(
            (len(sequences), maxlen), self.value, dtype=self.dtype
        )

        for i, seq in enumerate(sequences):
            if self.truncating == "pre":
                trunc = seq[-maxlen:]
            elif self.truncating == "post":
                trunc = seq[:maxlen]
            else:
                raise ValueError(f'Truncating type "{self.truncating}" not understood')

            if self.padding == "pre":
                padded_sequences[i, -len(trunc) :] = trunc
            elif self.padding == "post":
                padded_sequences[i, : len(trunc)] = trunc
            else:
                raise ValueError(f'Padding type "{self.padding}" not understood')

        return padded_sequences

    def texts_to_pad_sequences(self, text: list):
        """
        Converts a list of text strings into padded sequences.

        Parameters
        ----------
        text : list
            The list of text strings to convert.

        Returns
        -------
        np.ndarray
            The padded sequences.
        """
        sequences = self.texts_to_sequences(text)
        return self.sequences_to_pad(sequences)

    @property
    def word_index(self):
        """
        Returns the word-to-index dictionary.

        Returns
        -------
        dict
            The word-to-index dictionary.
        """
        return self.__str_to_int

    @property
    def index_word(self):
        """
        Returns the index-to-word dictionary.

        Returns
        -------
        dict
            The index-to-word dictionary.
        """
        return self.__int_to_str



def count_words(
    data: Union[List[str], List[Tuple[str]]],
    min_count: Optional[int] = None,
    return_dataframe: bool = False,
):
    """
    Count the occurrences of words in a list of tokenized texts or strings.

    Parameters
    ----------
    data : Union[List[str], List[Tuple[str]]]
        List of tokenized texts (list of tuples or list of strings).
    min_count : Optional[int], optional
        Minimum count threshold for words to be included in the result, by default None.
    return_dataframe : bool, optional
        Whether to return the result as a pandas DataFrame, by default False.

    Returns
    -------
    Union[dict, pd.DataFrame]
        If return_dataframe is False, returns a dictionary where keys are words and values are counts.
        If return_dataframe is True, returns a DataFrame with columns "words" and "counts".
    """
    from collections import Counter

    if not any(isinstance(x, (list, tuple)) for x in data):
        data = word_tokenize(data)

    data = [element for row in data for element in row]

    counter_words = Counter(data)
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


__all__ = [
    "f_remove_punctuation",
    "remove_punctuation",
    "f_remove_digits",
    "remove_digits",
    "f_remove_url",
    "remove_url",
    "f_remove_emoji",
    "remove_emoji",
    "f_remove_non_latin",
    "remove_non_latin",
    "F_TEXT_CLEAN",
    "lower_text",
    "f_regex_word_tokenize",
    "f_word_tokenize",
    "word_tokenize",
    "token_to_str",
    "find_duplicates",
    "text_clean",
    "get_stopwords_en",
    "get_stopwords_idn",
    "f_stopwords",
    "remove_stopwords",
    "f_normalize",
    "normalize",
    "f_lemmatization_en",
    "f_lemmatization_idn",
    "lemmatization",
    "f_stemming_en",
    "f_stemming_idn",
    "stemming",
    "text_manipulation",
    "Tokenizer",
    "count_words",
]
