import string
import re
from typing import Union, List, Optional, Pattern, Set, Tuple, Literal
from nltk.corpus import stopwords as st
from functools import lru_cache


@lru_cache(maxsize=None)
def f_remove_punctuation(text: str, punctuation: str = string.punctuation) -> str:
    """
    Remove punctuation from a single text string.

    Args:
        text : The input text from which to remove punctuation.
        punctuation : The punctuation characters to remove, by default string.punctuation.

    Returns:
        str: The text with punctuation removed.
    """
    translation = str.maketrans(punctuation, " " * len(punctuation))
    return text.translate(translation)


def remove_punctuation(
    data: List[str], punctuation: str = string.punctuation
) -> List[str]:
    """
    Remove punctuation from a list of texts.

    Args:
        data : List of texts.
        punctuation : The punctuation characters to remove, by default string.punctuation.

    Returns:
        List[str] : List of texts without punctuation.
    """
    return [f_remove_punctuation(text, punctuation) for text in data]


@lru_cache(maxsize=None)
def f_remove_digits(text: str) -> str:
    """
    Remove digits from a single text string.

    Args:
        text : The input text from which to remove digits.

    Returns:
        str: The text with digits removed.
    """
    return re.sub(r"\d+", "", text)


def remove_digits(data: List[str]) -> List[str]:
    """
    Remove digits from a list of texts.

    Args:
        data : List of texts.

    Returns:
        List[str]: List of texts without digits.
    """
    return [f_remove_digits(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_tag_html(text: str) -> str:
    pattern = re.compile("<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(pattern, " ", text)


def remove_tag_html(data: List[str]) -> List[str]:
    return [f_remove_tag_html(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_url(text) -> str:
    """
    Remove URLs from a single text string.

    Args:
        text : The input text from which to remove URLs.

    Returns:
        str: The text with URLs removed.
    """
    return re.compile(r"https?://\S+|www\.\S+").sub(r"", text)


def remove_url(data: List[str]) -> List[str]:
    """
    Remove URLs from a list of texts.

    Args:
        data : List of texts.

    Returns:
        List[str]: List of texts without URLs.
    """
    return [f_remove_url(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_emoji(text: str) -> str:
    """
    Remove emojis from a single text string.

    Args:
        text : The input text from which to remove emojis.

    Returns:
        str: The text with emojis removed.
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

    Args:
        data : List of texts.

    Returns:
        List[str]: List of texts without emojis.
    """
    return [f_remove_emoji(text) for text in data]


@lru_cache(maxsize=None)
def f_remove_non_latin(text: str) -> str:
    """
    Remove non-Latin characters from a single text string.

    Args:
        text : The input text from which to remove non-Latin characters.

    Returns:
        str: The text with non-Latin characters removed.
    """
    return re.sub(r"[^\x00-\x7F]+", "", text)


def remove_non_latin(data: List[str]) -> List[str]:
    """
    Remove non-Latin characters from a list of texts.

    Args:
        data : List of texts.

    Returns:
        List[str]: List of texts without non-Latin characters.
    """
    return [f_remove_non_latin(text) for text in data]


def f_remove_white_space(text: str) -> str:
    pattern = re.compile(r"\s+")
    return re.sub(pattern, " ", text).strip()


def remove_white_space(data: List[str]) -> List[str]:
    return [f_remove_white_space(text) for text in data]


def f_remove_one_chars(token: Tuple[str]):
    return (i for i in token if len(i) != 1)


def remove_one_chars(tokens: List[Tuple[str]]):
    if not any(isinstance(x, (list, tuple)) for x in tokens):
        raise ValueError("text data must be tokenized first")
    return [f_remove_one_chars(token) for token in tokens]


F_TEXT_CLEAN = [
    "remove_url",
    "remove_emoji",
    "remove_tag_html" "remove_punctuation",
    "remove_digits",
    "remove_non_latin",
]


def lower_text(data: List[str]) -> List[str]:
    return [x.lower() for x in data]


def f_regex_word_tokenize(text: str, pattern: Pattern[str]) -> Tuple[str]:
    """
    Tokenize a text string using a regex pattern.

    Args:
        text : The input text to tokenize.
        pattern : The regex pattern to use for tokenization.

    Returns:
        Tuple[str]: A tuple of tokens.
    """
    from nltk.tokenize import regexp_tokenize

    return tuple(regexp_tokenize(text=text, pattern=pattern))


def f_word_tokenize(text: str):
    """
    Tokenize a text string using NLTK's word_tokenize.

    Args:
        text : The input text to tokenize.

    Returns:
        Tuple[str]: A tuple of tokens.
    """
    from nltk.tokenize import word_tokenize as wt

    return tuple(wt(text))


def word_tokenize(
    data: List[str], pattern: Optional[Pattern[str]] = None
) -> List[Tuple[str, ...]]:
    """
    Tokenize a list of text strings.

    Args:
        data : List of texts to tokenize.
        pattern : The regex pattern to use for tokenization, by default None.

    Returns:
        List[Tuple[str, ...]]: List of tuples of tokens.
    """
    if pattern:
        return [f_regex_word_tokenize(x, pattern=pattern) for x in data]

    return [f_word_tokenize(x) for x in data]


def token_to_str(data: List[Tuple[str]]) -> List[str]:
    """
    Convert a list of token tuples to a list of strings.

    Args:
        data : List of token tuples.

    Returns:
        List[str]: List of strings joined from tokens.
    """
    return [" ".join(x) for x in data]


def find_duplicates(data: List[str]) -> dict:
    """
    Find duplicates in a list of strings and return their indices.

    Args:
        data : List of strings to check for duplicates.

    Returns:
        dict: A dictionary where keys are duplicate strings and values are lists of their indices.
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
    tag_html: bool = True,
    url: bool = True,
    non_latin: bool = True,
    white_space: bool = True,
    return_token: bool = False,
    return_dataframe: bool = False,
    verbose: bool = False,
    pattern: Optional[Pattern[str]] = None,
):
    """
    Clean a list of text data based on specified parameters.

    Args:
        data : List of strings to clean.
        punctuation : Punctuation characters to remove, by default string.punctuation.
        lower : Convert text to lowercase, by default True.
        digits : Remove digits from text, by default True.
        emoji : Remove emojis from text, by default True.
        duplicates : Remove duplicate entries from data, by default True.
        url : Remove URLs from text, by default True.
        non_latin : Remove non-Latin characters from text, by default True.
        return_token : Tokenize text and return tokens, by default False.
        return_dataframe : Return result as a Pandas DataFrame, by default False.
        verbose : Display progress using tqdm, by default False.
        pattern : Regex pattern for tokenization, by default None.

    Returns:
        List[str]: Cleaned list of text data.

    Raises:
        ValueError: If return_dataframe is True but lengths of data and cleaned data don't match.
    """
    from tqdm import tqdm

    def maybe_tqdm(iterable, desc):
        return tqdm(iterable, desc=desc, colour="cyan") if verbose else iterable

    idx = None
    if duplicates:
        idx = find_duplicates(data)
        if idx:
            sample = list(idx.values())[:5]
            print(f"Duplication found in data: {sample} ...")
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

    if tag_html:
        datac = remove_tag_html(maybe_tqdm(datac, "Removing Tag HTML"))

    if punctuation:
        datac = remove_punctuation(
            maybe_tqdm(datac, "Removing Punctuation"), punctuation=punctuation
        )

    if digits:
        datac = remove_digits(maybe_tqdm(datac, "Removing Digits"))

    if non_latin:
        datac = remove_non_latin(maybe_tqdm(datac, "Removing Non-Latin Characters"))

    if white_space:
        datac = remove_white_space(maybe_tqdm(datac, "Removing White Space"))

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

    Returns:
        List[str]: List of English stopwords.
    """
    return st.words("english")


def get_stopwords_idn() -> List[str]:
    """
    Get Indonesian stopwords from NLTK corpus.

    Returns:
        List[str]
            List of Indonesian stopwords.
    """
    return st.words("indonesian")


def f_stopwords(
    token: Union[Tuple[str], str], stopwords: Set[str], return_token: bool = False
) -> Union[Tuple[str, ...], str]:
    """
    Remove stopwords from a token or tuple of tokens.

    Args:
        token : Token or tuple of tokens to filter.
        stopwords : Set of stopwords to filter out.
        return_token : Whether to return tokens (True) or a joined string (False), by default False.

    Returns:
        Union[Tuple[str, ...], str]: Filtered tokens or joined string without stopwords.
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
    stop_one_chars: bool = True,
    return_token: bool = True,
    verbose: bool = False,
) -> Union[List[str], List[Tuple[str]]]:
    """
    Remove stopwords from a list of tokenized texts.

    Args:
        tokens : List of tokenized texts, where each item can be a tuple or list of tokens.
        lang : Language code for stopwords ('english' or 'indonesian').
        stopwords : List of additional stopwords to remove, by default None.
        additional : List of additional stopwords to add, by default None.
        return_token : Whether to return tokens (True) or joined strings (False), by default True.
        verbose : Whether to display progress bar, by default False.

    Returns:
        Union[List[str], List[Tuple[str]]]: List of tokenized texts with stopwords removed.
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

    if stop_one_chars:
        tokens = remove_one_chars(tokens)

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


def f_normalize(
    token: Union[Tuple[str], str], norm_words: dict, return_token: bool = False
) -> Union[Tuple[str, ...], str]:
    """
    Normalize tokens using a dictionary of normalization mappings.

    Args:
        token : Token or tuple of tokens to normalize.
        norm_words : Dictionary mapping tokens to their normalized forms.
        return_token : Whether to return tokens (True) or a joined string (False), by default False.

    Returns:
        Union[Tuple[str, ...], str]: Normalized tokens or joined string of normalized tokens.
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

    Args:
        tokens : List of tokenized texts, where each item can be a tuple or list of tokens.
        norm_words : Dictionary mapping tokens to their normalized forms.
        return_token : Whether to return tokens (True) or joined strings (False), by default True.
        verbose : Whether to display progress bar, by default False.

    Returns:
        List[Union[str, Tuple[str]]]: List of normalized texts.
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

    Args:
        text : Input text to lemmatize.
        return_token : Whether to return tokens (True) or joined string (False), by default False.

    Returns:
        Union[Tuple[str], str]:
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

    Args:
        token : Token or tuple of tokens to lemmatize.
        return_token : Whether to return tokens (True) or joined string (False), by default False.

    Returns:
        Union[Tuple[str, ...], str]:
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

    Args:
        text : Input text to stem.
        return_token : Whether to return tokens (True) or joined string (False), by default False.

    Returns:
        Union[Tuple[str, ...], str]:
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

    Args:
        text : Input text to stem.
        return_token : Whether to return tokens (True) or joined string (False), by default False.

    Returns:
        Union[Tuple[str, ...], str]:
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

    Args:
        data : List of tokenized texts, where each item can be a tuple or list of tokens.
        lang : Language code for lemmatization ('indonesian' or 'english').
        return_token : Whether to return tokens (True) or joined string (False), by default False.
        verbose : Whether to display progress bar, by default False.

    Returns:
        List[Union[Tuple[str], str]]:
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
            lemmatized = tuple(__lemma_word(word) for word in sentence)
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

    Args:
        data : List of tokenized texts, where each item is a tuple of tokens.
        lang : Language code for stemming ('indonesian' or 'english').
        return_token : Whether to return tokens (True) or joined string (False), by default False.
        verbose : Whether to display a progress bar, by default False.

    Returns:
        List[str]:
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

    Args:
        tokens : List of tokenized texts, where each item is a tuple of tokens.
        lang : Language code for text manipulation ('indonesian' or 'english').
        stopwords : List of stopwords or True to use default stopwords for the specified language, by default False.
        stem : Whether to perform stemming (True) or lemmatization (False), by default False.
        return_dataframe : Whether to return results as a DataFrame, by default False.
        norm_words : Dictionary of normalization words, by default None.
        return_token : Whether to return tokens (True) or joined string (False), by default False.
        additional : Additional stopwords to include, by default None.
        verbose : Whether to display progress bars for preprocessing steps, by default False.

    Returns:
        Union[List[str], pd.DataFrame]:
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
                datac,
                lang=lang,
                return_token=True,
                verbose=verbose,
                additional=additional,
            )
        else:
            datac = remove_stopwords(
                datac,
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
]
