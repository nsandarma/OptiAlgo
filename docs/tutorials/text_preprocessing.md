# Text Preprocessing
The `text_preprocessing` module can help you perform text cleaning & manipulation quickly and efficiently 🫠.

## Text Cleaning
You can clean text easily, just use the `text_clean` function.

**Note :** for the arguments in function `text_clean`, see [text_clean](https://nsandarma.github.io/OptiAlgo/utils/text_preprocessing/#optialgo.text_clean)
```python exec="true" source="above" result="python" session="text_preprocessing"
from optialgo.text_preprocessing import text_clean
dataset = ["I told you not to feed that dog!","There are 25 students in my math class this semester.",
"For more information about the project, please visit https://github.com/nsandarma/OptiAlgo","My favorite Japanese word is 'ありがとう' which means 'thank you'.","I'm so excited for the weekend! 😄"
]
dataset_clean = text_clean(data=dataset) # if returns is tokens, set parameters `return_token` is `True`
print(dataset_clean)
```
### Remove Punctuation
```python exec="true" source="above" result="python" session="text_preprocessing"
from optialgo.text_preprocessing import *
data = ["I told you not to feed that dog!"]
print(remove_punctuation(data))
```
**Note:** 
use `f_remove_punctuation` if the data is `str` / single data
### Remove Digits
```python exec="true" source="above" result="python" session="text_preprocessing"
data = ["There are 25 students in my math class this semester."]
print(remove_digits(data))
```
**Note:** 
use `f_remove_digits` if the data is `str` / single data

### Remove URL
```python exec="true" source="above" result="python" session="text_preprocessing"
data = ["For more information about the project, please visit https://github.com/nsandarma/OptiAlgo"]
print(remove_url(data))
```
**Note:** 
use `f_remove_url` if the data is `str` / single data

### Remove chars-non-latin
```python exec="true" source="above" result="python" session="text_preprocessing"
data = ["My favorite Japanese word is 'ありがとう' which means 'thank you'."]
print(remove_non_latin(data))
```
**Note:** 
use `f_remove_non_latin` if the data is `str` / single data

### Remove Emoji
```python exec="true" source="above" session="text_preprocessing" result="python"
data = ["I'm so excited for the weekend! 😄"]
print(remove_emoji(data))
```
**Note:** 
use `f_remove_emoji` if the data is `str` / single data

## Text Manipulation
After cleaning the text, you can manipulate the text by using the `text_manipulation` function 🥶 .

see : [text_manipulation](https://nsandarma.github.io/OptiAlgo/utils/text_preprocessing/#optialgo.text_manipulation)
```python exec="true" source="above" session="text_preprocessing" result="python"
# before performing `text_manipulation` the data must be tokenized first
print(text_manipulation(tokens=word_tokenize(dataset_clean),lang="english")) # lang : ["indonesian","english"]
```

### Word Tokenize
```python exec="true" source="above" result="python" session="text_preprocessing"
tokens = word_tokenize(dataset_clean)
print(tokens)
```
**Note:** 
use `f_word_tokenize` if the data is `str` / single data *or* `f_regex_word_tokenize` with regex.

### Token To String
```python exec="true" source="above" session="text_preprocessing" result="python"
print(token_to_str(tokens))
```

### Word Normalization
```python exec="true" source="above" session="text_preprocessing" result="python"
norm_words = {"dog":"cat"}
print(normalize(tokens,norm_words))
```
**Note:** 
use `f_word_normalize` if the data is `str` / single data

### Stopwords Removal
```python exec="true" source="above" session="text_preprocessing" result="python"
print(remove_stopwords(tokens,lang="english"))

```
**You can also add additional stopwords :**

```python exec="true" source="above" session="text_preprocessing" result="python"
additional_st = ["dog"]
print(remove_stopwords(tokens,lang="english",additional=additional_st))
```

### Lemmatization
```python exec="true" source="above" session="text_preprocessing" result="python"
print(lemmatization(tokens,lang="english"))
```

**Note:** 
use `f_lemmatization_en` if the data is `str` / single data, `f_lemmatization_idn` for Indonesian text

### Stemming
```python exec="true" source="above" session="text_preprocessing" result="python"
print(stemming(tokens,lang="english"))

```

**Note:** 
use `f_stemming_en` if the data is `str` / single data, `f_stemming_idn` for Indonesian text


## Tokenizer
If you need a tokenizer to prepare for deep learning, optialgo already provides it 🫵

see : [tokenizer](https://nsandarma.github.io/OptiAlgo/utils/text_preprocessing/?h=tokenizer#optialgo.Tokenizer)
```python exec="true" source="above" session="text_preprocessing" result="python"
tokenizer = Tokenizer()
tokenizer.fit(dataset_clean)
print(tokenizer.word_index)
```
### Count Words

```python exec="true" source="above" session="text_preprocessing" result="python"
print(count_words(dataset_clean))
```

### Texts to Sequences
```python exec="true" source="above" session="text_preprocessing" result="python"
sequences = tokenizer.texts_to_sequences(dataset_clean)
print(sequences)
```

### Texts to Pad Sequences
```python exec="true" source="above" session="text_preprocessing" result="python"
sequences = tokenizer.texts_to_pad_sequences(dataset_clean)
print(sequences)
```

### Sequences/Pad to Texts

```python exec="true" source="above" session="text_preprocessing" result="python"
print(tokenizer.sequences_to_texts(sequences))
```