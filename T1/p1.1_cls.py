import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics



data_set = load_files(r"C:\Users\Ashkan\Documents\Concordia\2021\fall\COMP 472\Projects\P1\BBC",encoding='latin1')

X, y = data_set.data,data_set.target

vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X)


vocab = vectorizer.get_feature_names()

print("Total number of word-tokens: ",X_train_counts.getnnz())
print("size of vocabulary: ",len(vocab))


tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

print(X_train_tfidf.toarray())

X_train, X_test, y_train, y_test = train_test_split(X_train_tfidf, y, test_size=0.2, random_state=None)


classifiler = MultinomialNB(alpha=0.9).fit(X_train,y_train)


predicted = classifiler.predict(X_test)

cnfmatrix = metrics.confusion_matrix(y_test, predicted)
print(cnfmatrix)

clf_report = metrics.classification_report(y_test, predicted,target_names=data_set.target_names)

print(clf_report)





