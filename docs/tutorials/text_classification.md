# Text Classification
```python exec="true" source="above" result="dataframe" session="text_classification"
import pandas as pd
from optialgo import text_clean,text_manipulation,TextDataset,Classification

df = pd.read_csv("dataset_ex/IMDB Dataset.csv")[:200]
train = df.sample(190)
test = df.sample(10)

print(train)
```
## Preprocessing
```python exec="true" source="above" result="dataframe" session="text_classification"
texts  = train['review'].tolist()

# Text clean (remove : punctuation,digits,emoji,chars-non-latin,and urls)
texts = text_clean(texts,return_token=True)

# Text manipulation 
additional_st = ['s','b','br']
texts = text_manipulation(texts,lang="english",stopwords=True,additional=additional_st)
train['review'] = texts 
print(df)
```
## Make Dataset
```python exec="true" source="above" result="python" session="text_classification"
feature = "review"
target = "sentiment"

dataset = TextDataset(train)
dataset.fit(feature=feature, target=target, t="classification", vectorizer="tfidf")
print(dataset.train)
```

## Model
```python exec="true" source="above" result="python" session="text_classification"
clf = Classification(dataset)
print(clf)
```

## Predict Sentiment
```python exec="true" source="above" result="python" session="text_classification"
clf.set_model("Random Forest")
print(clf.predict(test[feature]))

```
