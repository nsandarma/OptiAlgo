import unittest
from optialgo import TextDataset, Classification
from optialgo.text_preprocessing import *
import random
from functools import lru_cache


class TestTextPreprocessing(unittest.TestCase):
    _texts = [
        word_tokenize(
            [
                "i told you not to feed that dog ",
                "there are  students in my math class this semester ",
                "for more information about the project  please visit ",
                "my favorite japanese word is which means  thank you  ",
                "i m so excited for the weekend",
            ]
        ),
        word_tokenize(
            [
                "saya tidak ingin makan nasi hari ini",
                "dia sangat rajin belajar setiap hari",
                "kami akan pergi ke pantai besok",
                "apakah kamu sudah menyelesaikan tugasmu?",
                "mereka bermain sepak bola di lapangan",
                "dia pergi ke pasar untuk membeli sayuran",
                "aku suka mendengarkan musik klasik",
                "kami berencana untuk liburan ke bali tahun depan",
                "mereka sedang menonton film di bioskop",
                "apakah kamu sudah membaca buku itu?",
            ]
        ),
    ]

    def test_remove_punc(self):
        import string

        punctuations = set(string.punctuation)
        texts = ["I told you not to feed that dog !", "do you love me ?"]
        after = remove_punctuation(texts)
        idx = random.randint(0, len(texts) - 1)
        self.assertEqual(f_remove_punctuation(texts[idx]), after[idx])
        self.assertFalse(
            any([any(char in punctuations for char in x) for x in after]),
            "found punc !",
        )
        self.assertIsInstance(after, list)

    def test_remove_digits(self):
        texts = [
            "There are 25 students in my math class this semester.",
            "ran 8 tests in 2.1s",
        ]
        after = remove_digits(texts)
        idx = random.randint(0, len(texts) - 1)
        self.assertEqual(f_remove_digits(texts[idx]), after[idx])
        self.assertFalse(
            any([any(char.isdigit() for char in x) for x in after]), "found digits !"
        )
        self.assertIsInstance(after, list)

    def test_remove_emoji(self):
        from emoji import is_emoji

        texts = ["I'm so excited for the weekend! 😄", "i love you 😍"]
        after = remove_emoji(texts)
        idx = random.randint(0, len(texts) - 1)
        self.assertEqual(f_remove_emoji(texts[idx]), after[idx])
        self.assertFalse(
            any([any(is_emoji(char) for char in x) for x in after]), "found emoji !"
        )
        self.assertIsInstance(after, list)

    def test_remove_char_non_latin(self):
        from alphabet_detector import AlphabetDetector

        ad = AlphabetDetector()
        texts = [
            "My favorite Japanese word is 'ありがとう'",
            "which means 'thank you'.",
        ]
        after = remove_non_latin(texts)
        idx = random.randint(0, len(texts) - 1)
        self.assertEqual(f_remove_non_latin(texts[idx]), after[idx])
        self.assertTrue(any(ad.is_latin(x) for x in after), "found char non latin !")
        self.assertIsInstance(after, list)

    def test_remove_url(self):
        import re

        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        texts = [
            "For more information about the project, please visit https://github.com/nsandarma/OptiAlgo",
            "google.com is website google",
        ]
        after = remove_url(texts)
        idx = random.randint(0, len(texts) - 1)
        self.assertEqual(f_remove_url(texts[idx]), after[idx])
        self.assertFalse([any(re.findall(regex, x) for x in after)][0], "found url")
        self.assertIsInstance(after, list)

    def test_word_tokenize(self):
        texts = ["I told you not to feed that dog !", "do you love me ?"]
        after = word_tokenize(texts)
        idx = random.randint(0, len(texts) - 1)
        self.assertTupleEqual(f_word_tokenize(texts[idx]), after[idx])
        self.assertTrue(any(isinstance(x, (list, tuple)) for x in after))

    def test_token_to_str(self):
        texts = [
            ("i", "told", "you", "not", "to", "feed", "that", "cat"),
            (
                "there",
                "are",
                "students",
                "in",
                "my",
                "math",
                "class",
                "this",
                "semester",
            ),
            (
                "for",
                "more",
                "information",
                "about",
                "the",
                "project",
                "please",
                "visit",
            ),
            (
                "my",
                "favorite",
                "japanese",
                "word",
                "is",
                "which",
                "means",
                "thank",
                "you",
            ),
            ("i", "m", "so", "excited", "for", "the", "weekend"),
        ]
        after = token_to_str(texts)
        self.assertTrue(any(isinstance(x, str) for x in after))

    def test_word_normalization(self):
        texts = word_tokenize(
            [
                "i told you not to feed that dog ",
                "there are  students in my math class this semester ",
                "for more information about the project  please visit ",
                "my favorite japanese word is which means  thank you  ",
                "i m so excited for the weekend",
            ]
        )

        norm_words = {"dog": "cat", "my": "mi"}
        after = normalize(texts, norm_words=norm_words)
        self.assertFalse(
            any([key in x for x in after for key in norm_words.keys()]),
            "norm_word on after!",
        )

    def test_stopword_removal(self):
        _st_additional = [["japanase", "please"], ["itu", "sayuran"]]
        _st = [["i", "not", "to"], ["dia", "saya", "ini"]]
        for lang in ["indonesian", "english"]:
            texts = self._texts[1] if lang == "indonesian" else self._texts[0]
            st_additional = (
                _st_additional[1] if lang == "indonesian" else _st_additional[0]
            )
            st = _st[1] if lang == "indonesian" else _st[0]

            st_default = (
                get_stopwords_en() if lang == "english" else get_stopwords_idn()
            )

            self.assertListEqual(
                [f_stopwords(text, st_default, return_token=True) for text in texts],
                remove_stopwords(texts, lang=lang, return_token=True),
            )
            self.assertListEqual(
                [f_stopwords(text, st_default, return_token=False) for text in texts],
                remove_stopwords(texts, lang=lang, return_token=False),
            )
            self.assertIs(
                (
                    any(
                        any(word in st for word in x)
                        for x in remove_stopwords(
                            texts, stopwords=st, lang=lang, return_token=True
                        )
                    )
                ),
                False,
                "st not removal",
            )
            self.assertIs(
                (
                    any(
                        any(word in st_additional for word in x)
                        for x in remove_stopwords(
                            texts,
                            additional=st_additional,
                            lang=lang,
                            return_token=True,
                        )
                    )
                ),
                False,
                "st_additional not removal",
            )

    def test_lemmatization(self):
        for lang in ["indonesian", "english"]:
            texts = self._texts[0] if lang == "english" else self._texts[1]
            if lang == "english":
                self.assertListEqual(
                    [f_lemmatization_en(text, return_token=False) for text in texts],
                    lemmatization(texts, lang=lang, return_token=False),
                )
                self.assertListEqual(
                    [f_lemmatization_en(text, return_token=True) for text in texts],
                    lemmatization(texts, lang=lang, return_token=True),
                )
            else:
                self.assertListEqual(
                    [
                        f_lemmatization_idn(text, return_token=True)
                        for text in token_to_str(texts)
                    ],
                    lemmatization(texts, lang=lang, return_token=True),
                )

    def test_stemming(self):
        for lang in ["indonesian", "english"]:
            texts = self._texts[0] if lang == "english" else self._texts[1]
            if lang == "english":
                self.assertListEqual(
                    [f_stemming_en(text, return_token=False) for text in texts],
                    stemming(texts, lang=lang, return_token=False),
                )
                self.assertListEqual(
                    [f_stemming_en(text, return_token=True) for text in texts],
                    stemming(texts, lang=lang, return_token=True),
                )
            else:
                self.assertListEqual(
                    [
                        f_stemming_idn(text, return_token=True)
                        for text in token_to_str(texts)
                    ],
                    stemming(texts, lang=lang, return_token=True),
                )
