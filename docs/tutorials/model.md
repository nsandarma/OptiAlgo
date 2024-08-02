# **Model**
You can perform classification very easily by simply entering a pre-made `dataset`🫨. 

```python exec="true" source="above" result="python" session="dataset"
from optialgo import Classification

clf = Classification(dataset=dataset)
print(clf)

```
## Compare Models
***Notes*** :
```
output : 
- table : only displays the output table
- dataframe : returns the dataframe 
- only_accuracy : only returns the accuracy score

train_val : using train test split or using cross validation 
verbose : displaying progress or not
```

```python exec="true" source="above" result="python" session="dataset"
print(clf.compare_model(output="dataframe",train_val=True,verbose=True))
```

## Set Algorithm
Having found the `compare_model` score, you can now determine what algorithm to use 😉.
```
actually, you can specify the algorithm to be used at the beginning.
clf = Classification(dataset=dataset,algorithm="Logistic Regression")
```
```python exec="true" source="above" session="dataset"
clf.set_model("Logistic Regression")
```

## Find Best HyperParameters
```python exec="true" source="above" result="python" session="dataset"
print(clf.get_params_from_model)
```
for an example of this case, see [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)


you can change the parameter above, with the following estimation :

```python exec="true" source="above" result="python" session="dataset"
params = {"C":[1.0,1.2,1.5,2.0],"solver":["lbfgs", "liblinear", "newton-cg", "newton-cholesky", "sag", "saga"]}
best_score,best_params = clf.find_best_params(params)
print(f"best_score : {best_score} | best_params : {best_params}")
```

## Tuning HyperParameter
after finding the best hyperparameter, you can tune the hyperparameter.
```python exec="true" source="above" session="dataset"
clf.set_params(best_params)
```

## Prediction
After determining the algorithm, you can make predictions for the actual test data 🤫.
```python exec="true" source="above" session="dataset" result="python"
new_data = [[22,"F","HIGH","HIGH",23.1]]
print(clf.predict(new_data))
```

**Note:**
*For the regression case, you can simply call the `Regression` class, but first make sure `t` in your dataset = `regression`.*

```python exec="true" source="above" session="regression" result="python"
import pandas as pd
from optialgo import Dataset,Regression

df = pd.read_csv("dataset_ex/Housing.csv") # Regression Dataset
dataset = Dataset(df)
features = df.columns[1:] # ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']
target = "price"
dataset.fit(features=features,target=target,t="regression")

reg = Regression(dataset)
print(reg)
```

