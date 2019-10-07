#!/usr/bin/env python
# coding: utf-8

# In[147]:


import pandas as pd
import numpy as np
df= pd.read_csv("K:\Fall 2019\MLF\Assignments\HW6\ccdefault.csv")
X = df.iloc[:,1:23]
y = df.iloc[:,24] 

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score

train_score=[]
test_score=[]
k_range=np.arange(1,11)
for k in k_range:
    X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=k)
    dt= DecisionTreeClassifier(max_depth=6, random_state=k)
    dt.fit(X_train, y_train)
    y_train_pred=dt.predict(X_train)
    y_test_pred=dt.predict(X_test)
    score=accuracy_score(y_train, y_train_pred)
    score1= accuracy_score(y_test, y_test_pred)
    train_score.append(score)
    test_score.append(score1)
    print('Random-state:', k, 'Train Accuracy: {:.3f}'.format(score), 'Test Accuracy: {:.3f}'.format(score1))
    
print('Mean ', 'Train: {:.3f} '.format(np.mean(train_score)), 'Test: {:.3f}'.format(np.mean(test_score)))
print('Standard deviation ', 'Train: {:.3f} '.format(np.std(train_score)), 'Test: {:.3f}'.format(np.std(test_score)))

#Cross Validation
from sklearn.model_selection import cross_val_score
X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.1, random_state=42)
dt=DecisionTreeClassifier(max_depth=6, random_state=42)
dt.fit(X_train, y_train)
score2 = cross_val_score(dt, X_train, y_train, cv=10)
print("Stratified K-Fold Training Accuracy score: ")
print(score2)
score3 = cross_val_score(dt, X_test, y_test, cv=10)
print("Stratified K-Fold Test Accuracy score: ")
print(score3)
print("Stratified K-Fold Mean Accuracy score")
print("Train: {:.3f}".format(np.mean(score2)), "Test: {:.3f}".format(np.mean(score3)))
print("Standard Deviation of Stratified K-Fold Accuracy score")
print("Train: {:.3f}".format(np.std(score2)), "Test: {:.3f}".format(np.std(score3)))


print("My name is Khavya Chandrasekaran")
print("My NetID is: khavyac2")
print("I hereby certify that I have read the University policy on Academic Integrity and that I am not in violation.")

# In[ ]:




