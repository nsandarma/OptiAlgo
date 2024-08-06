#!/usr/bin/env python3
import os
import sys

from optialgo.tokenizer import Tokenizer
from optialgo import Classification,TextDataset,random_split
from optialgo.text_preprocessing import text_clean,text_manipulation
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score


def preprocessing(texts,verbose):

  texts = text_clean(texts,return_token=True,verbose=verbose)

  norm_dict = {"yg":"yang","syg":"sayang","sdh":"sudah","sy":"saya"}
  texts = text_manipulation(texts,lang="indonesian",return_token=False,stem=False,verbose=verbose,return_dataframe=False,stopwords=True,norm_words=norm_dict)
  return texts


if __name__ == "__main__":

  df = pd.read_csv("dataset_ex/example_idn.csv")
  feature = "ulasan"
  target = "rating"

  X = df[feature].values
  y = df[target].values
  X_train,y_train,X_test,y_test = random_split(X,y,train_size=0.8,random_state=32)
  df_train = pd.DataFrame({feature:X_train.reshape(-1),target:y_train.reshape(-1)})

  texts = df_train[feature].tolist()
  texts = preprocessing(texts,verbose=1)

  df_train = pd.DataFrame({feature:texts,target:df_train[target].tolist()})
  dataset = TextDataset(df_train)

  tfidf = TfidfVectorizer(max_features=1000)
  dataset.fit(feature=feature,target=target,t="classification",vectorizer=tfidf)

  clf = Classification(dataset)
  clf.compare_model(output="table",train_val=True,verbose=True)
  clf.set_model("Random Forest")

  texts = preprocessing(X_test.reshape(-1),verbose=0)

  pred = np.array(clf.predict(texts)).astype(str)
  y_test = y_test.reshape(-1).astype(str)
  score = accuracy_score(y_true=y_test,y_pred = pred)
  print("accuracy of data test : ",score)
  # result : 0.74 
