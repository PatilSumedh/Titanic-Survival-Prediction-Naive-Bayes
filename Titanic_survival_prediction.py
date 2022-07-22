#Importing required libraries

import pandas as pd #useful for loading the dataset
import numpy as np #to perform array
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Summarize Dataset

print(dataset.shape)
print(dataset.head(5))

#Mapping Text Data to Binary Value

income_set = set(dataset['Sex'])
dataset['Sex'] = dataset['Sex'].map({'female': 0, 'male': 1}).astype(int)
print(dataset.head)

#Segregate Dataset into X(Input/IndependentVariable) & Y(Output/DependentVariable)

X = dataset.drop('Survived',axis='columns') #X-->Input(IndependentVariable)
X

Y = dataset.Survived #Y --> Output(DependentVariable)
Y

X.columns[X.isna().any()]

X.Age = X.Age.fillna(X.Age.mean())

#Test again to check any na value

X.columns[X.isna().any()]

#Splitting Dataset into Train & Test data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25,random_state =0)

#Training

clf = GaussianNB()
clf.fit(X_train, Y_train)

# Accuracy for all Train & Test Data.

pred_tndata = clf.predict(X_train) #prediction on training data
tn_accuracy = accuracy_score(Y_train,pred_tndata)#checking accuracy of training data
print("Accuracy on training data :",tn_accuracy*100,"%")

pred_tsdata =clf.predict(X_test) #prediction on testing data
ts_accuracy = accuracy_score(Y_test,pred_tsdata)
print("Accuracy on testing data :",ts_accuracy*100,"%")

#Predicting, wheather Person Survived or Not

pclassNo = int(input("Enter Person's Pclass number: ")) #Enter Class i.e 1,2 or 3
gender = int(input("Enter Person's Gender: ")) #Enter 0 for Female & 1 for Male
age = int(input("Enter Person's Age: "))
fare = float(input("Enter Person's Fare: ")) #Enter fare in decimal eg. 7.25 ,55.20 ,etc
person = [[pclassNo,gender,age,fare]]
prediction = clf.predict(person)
print(prediction)

if prediction == 1:
    print("Person might be Survived")
else:
    print("Person might not be Survived")
