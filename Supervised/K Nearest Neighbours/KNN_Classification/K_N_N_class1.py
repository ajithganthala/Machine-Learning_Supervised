"""Here will be predicting the admission
based upon the score of gre,gmat,gpa work experience
by K Nearest Neighbour algorithm"""

#importing libraries
import pandas as pd #Importing pandas libraray for data preprocessing,cleaning and puting data to dataset.
import numpy as np #Importing numoy for numerical python
from sklearn.model_selection import train_test_split #importing the train,test and split class from model
from sklearn import metrics #importing metrics library

#Reading and putting data to dataframe.
df = pd.read_csv('gre.csv')

#We are storing the attributes to X and Y variable by usinf slicing
X = df.iloc[:,:-1]
y = df.iloc[:,-1]

#Splitting the data into testing and training 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=0)

#importing KNNclassifier library
from sklearn.neighbors import KNeighborsClassifier

#creating object for knn library
n=KNeighborsClassifier(n_neighbors=5,metric='euclidean')

# Including the values of X and Y to KNN algorithm
n.fit(X_train,y_train)

#Predicting the result of admission with data in X test
y_pred=n.predict(X_test)

#printing the predicted values
print(y_pred)

#Finding the accuracy of predicted value(Y_pred) by comparing with actual result(y_test).
print('accuracy',metrics.accuracy_score(y_test,y_pred))


#Testing the algorithm by giving input values.
"""a = int(input('enter the score of gmat'))
b = float(input('enter the score of gpa'))
c = int(input('enter work experience'))

testX = [[a,b,c]]
testp = n.predict(testX)
print('prediction',testp)"""

