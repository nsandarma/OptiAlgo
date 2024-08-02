import unittest 
from optialgo import TextDataset,Classification
from optialgo.text_preprocessing import *
import random

class TestTextPreprocessing(unittest.TestCase):
    def test_remove_punc(self):
        import string

        punctuations = set(string.punctuation)
        texts = ['I told you not to feed that dog !','do you love me ?']
        after = remove_punctuation(texts)
        idx = random.randint(0,len(texts) - 1)
        self.assertEqual(f_remove_punctuation(texts[idx]),after[idx])
        self.assertFalse(any([any(char in punctuations for char in x) for x in after]),"found punc !")
        self.assertIsInstance(after,list)
    
    def test_remove_digits(self):
        texts = ["There are 25 students in my math class this semester.","ran 8 tests in 2.1s"]
        after  = remove_digits(texts)
        idx = random.randint(0,len(texts)-1)
        self.assertEqual(f_remove_digits(texts[idx]),after[idx])
        self.assertFalse(any([any(char.isdigit() for char in x) for x in after]), "found digits !")
        self.assertIsInstance(after,list)

    def test_remove_emoji(self):
        from emoji import is_emoji
        texts = ["I'm so excited for the weekend! 😄","i love you 😍"] 
        after = remove_emoji(texts)
        idx = random.randint(0,len(texts)-1)
        self.assertEqual(f_remove_emoji(texts[idx]),after[idx])
        self.assertFalse(any([any(is_emoji(char) for char in x) for x in after]),"found emoji !")
        self.assertIsInstance(after,list)
        
    def test_remove_char_non_latin(self):
        from alphabet_detector import AlphabetDetector
        ad = AlphabetDetector()
        texts = ["My favorite Japanese word is 'ありがとう'","which means 'thank you'."]
        after = remove_non_latin(texts)
        idx = random.randint(0,len(texts)-1)
        self.assertEqual(f_remove_non_latin(texts[idx]),after[idx])
        self.assertTrue(any(ad.is_latin(x) for x in after),"found char non latin !")
        self.assertIsInstance(after,list)
    
    def test_remove_url(self):
        import re
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        texts = ["For more information about the project, please visit https://github.com/nsandarma/OptiAlgo","google.com is website google"]
        after = remove_url(texts)
        idx = random.randint(0,len(texts)-1)
        self.assertEqual(f_remove_url(texts[idx]),after[idx])
        self.assertFalse([any(re.findall(regex,x) for x in after)][0],"found url")
        self.assertIsInstance(after,list)

    def test_word_tokenize(self):
        texts = ['I told you not to feed that dog !','do you love me ?']
        after = word_tokenize(texts) 
        idx = random.randint(0,len(texts)-1)
        self.assertTupleEqual(f_word_tokenize(texts[idx]),after[idx])
        self.assertTrue(any(isinstance(x,(list,tuple)) for x in after ))

    def test_token_to_str(self):
        texts = [('i', 'told', 'you', 'not', 'to', 'feed', 'that', 'cat'), 
                 ('there', 'are', 'students', 'in', 'my', 'math', 'class', 'this', 'semester'), 
                 ('for', 'more', 'information', 'about', 'the', 'project', 'please', 'visit'), 
                 ('my', 'favorite', 'japanese', 'word', 'is', 'which', 'means', 'thank', 'you'), 
                 ('i', 'm', 'so', 'excited', 'for', 'the', 'weekend')]
        after = token_to_str(texts)
        self.assertTrue(any(isinstance(x,str) for x in after ))
    
    def test_word_normalization(self):
        texts = word_tokenize(['i told you not to feed that dog ', 
                'there are  students in my math class this semester ', 
                'for more information about the project  please visit ', 
                'my favorite japanese word is which means  thank you  ', 
                'i m so excited for the weekend'])
        
        norm_words = {"dog":"cat","my":"mi"}
        after = normalize(texts,norm_words=norm_words)
        self.assertFalse(any([key in x for x in after for key in norm_words.keys()]),"norm_word on after!")
    
    def test_stopword_removal(self):
        texts = word_tokenize(['i told you not to feed that dog ', 
                'there are  students in my math class this semester ', 
                'for more information about the project  please visit ', 
                'my favorite japanese word is which means  thank you  ', 
                'i m so excited for the weekend'])
        st_additional = ["weekend","math"]
        st = ["i","m","so"]
        lang = "english"

        if lang == "english":
            st_default = get_stopwords_en()
        else :
            st_default = get_stopwords_idn()

        self.assertListEqual([f_stopwords(text,st_default,return_token=True) for text in texts],remove_stopwords(texts,lang=lang,return_token=True))
        self.assertListEqual([f_stopwords(text,st_default,return_token=False) for text in texts],remove_stopwords(texts,lang=lang,return_token=False))
        self.assertIs((any(any(word in st for word in x)  for x in remove_stopwords(texts,stopwords=st, lang = lang, return_token=True))),False,"st not removal")
        self.assertIs((any(any(word in st_additional for word in x)  for x in remove_stopwords(texts,additional=st_additional, lang = lang, return_token=True))),False,"st_additional not removal")
    


        




    


    
        








    




        
        
        



