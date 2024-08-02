# **Make Dataset**
the assumption is that `Dataset` is like pandas dataframe. you don't need preprocessing like doing `ColumnTransformer`, everything is already done in `Dataset` 🫡. 
```python exec="true" source="above" result="python" session="dataset"
from optialgo import Dataset,TextDataset
import pandas as pd

df = pd.read_csv("dataset_ex/drug200.csv")

dataset = Dataset(df,test_size=0.2) # if you want the data to be normalized then set `norm=True`
print(dataset.dataframe)
```
*Fit features and target* :

```python exec="true" source="above" result="python" session="dataset"
features = df.columns[:-1] # ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
target = "Drug" # column target

dataset.fit(features=features,target=target,t="classification") 
print(dataset)
```

after fit, you get train set and test set.

```python exec="true" source="above" result="python" session="dataset"
print(dataset.train)
```

```python exec="true" source="above" result="python" session="dataset"
print(dataset.test)
```

## Customization `ColumnTransform`
* `target_mean` : (https://hackernoon.com/the-concept-behind-mean-target-encoding-in-ai-and-ml)
* `one_hot` : (https://medium.com/@WojtekFulmyk/one-hot-encoding-a-brief-explanation-8c5daec395e3)
* `ordinal` : (https://medium.com/@WojtekFulmyk/ordinal-encoding-a-brief-explanation-a29cf374dbc1)
```python exec="true" source="above" result="python" session="dataset"
cat_encoders = {"ordinal":["BP","Cholesterol"],"one_hot":["Sex"]}
dataset.fit(features=features,target=target,t="classification",encoder=cat_encoders)
print(dataset.train)
```

## Text Dataset
```python exec="true" source="above" result="python"
import pandas as pd
from optialgo import TextDataset

df_text = pd.read_csv("dataset_ex/IMDB Dataset.csv")[:10] # for example
dataset = TextDataset(df_text,test_size=0.1)

feature = "review" # the text field that will be used as a feature
target = "sentiment"

dataset.fit(feature=feature,target=target,t="classification",vectorizer='tfidf')
print(dataset.train)
```
