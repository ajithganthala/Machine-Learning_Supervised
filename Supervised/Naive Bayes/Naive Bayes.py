# import dependencies
import numpy as np
import pandas as pd

# Imporitng Data
dt = pd.read_csv('weather.csv')
print(dt.head(60))


dt['play'],play_conform=pd.factorize(dt['play'])
dt['outlook'],_=pd.factorize(dt['outlook'])
dt['temperature'],_=pd.factorize(dt['temperature'])
dt['humidity'],_=pd.factorize(dt['humidity'])
dt['windy'],_=pd.factorize(dt['windy'])

print(dt.head(60))

X = dt.iloc[:,0:-1] # X is the features in our dataset
y = dt.iloc[:,-1]   # y is the Labels in our dataset

print(X.head())

print(y.head())

# divide the dataset in train test using scikit learn
# now the model will train in training dataset and then we will use test dataset to predict its accuracy

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=1) 

# now preparing our model as per Gaussian Naive Bayesian

from sklearn.naive_bayes import GaussianNB

model = GaussianNB().fit(X_train, y_train) #fitting our model

y_pred = model.predict(X_test) #now predicting our model to our test dataset

from sklearn.metrics import accuracy_score

# now calculating that how much accurate our model is with comparing our predicted values and y_test values
accuracy_score = accuracy_score(y_test, y_pred) 
print (accuracy_score)

from sklearn.metrics import accuracy_score, confusion_matrix
conf_mat = confusion_matrix(y_test,y_pred)
print(conf_mat)

true_positive = conf_mat[0][0]
false_positive = conf_mat[0][1]
false_negative = conf_mat[1][0]
true_negative = conf_mat[1][1]

# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
print(Accuracy)

# Precison
Precision = true_positive/(true_positive+false_positive)
print(Precision)

# Recall
Recall = true_positive/(true_positive+false_negative)
print(Recall)

new = pd.DataFrame()

# Create some feature values for this single row
new['outlook'] = [0]
new['temperature'] = [2]
new['humidity'] = [0]
new['windy'] = [1]

print(new)

predicted_y = model.predict(new)

print (predicted_y)



