# Home

## `optialgo` and `scikit-learn`
If you are used to modeling with `scikit-learn`, congratulations 🤩 you will find optialgo easy to use!

## Main Features
```
1. Data Prepration
2. Data Preprocessing
3. Text Preprocessing (2x Faster)
3. Comparing Model
4. Set Model
5. Prediction
6. HyperParameter Tuning
```
## Installation

**Before installing OptiAlgo, it is recommended to create an environment first.**

```console
pip install optialgo
```
or

```console
pip install git+https://github.com/nsandarma/OptiAlgo.git
```

*and for text preprocessing*

```console
>>> import nltk
>>> nltk.download('all')
```

## Overview
```python
import pandas as pd
from optialgo import Dataset, Classification

df = pd.read_csv('dataset_ex/drug200.csv')
features = ['Age','Sex','BP','Cholesterol',"Na_to_K"]
target = 'Drug'

dataset = Dataset(dataframe=df)
dataset.fit(features=features,target=target)

clf = Classification()
result = clf.compare_model(output='table',train_val=True)
print(result)
```

![image](https://raw.githubusercontent.com/nsandarma/OptiAlgo/master/images/result.png)



