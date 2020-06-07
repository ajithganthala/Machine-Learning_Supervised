# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder

# Importing the dataset
df = pd.read_csv("cracow_apartments.csv")
print(df.head())

X= df.iloc[:,0:3].values
y = df.iloc[:,3:].values
#print(X)
#print(y)
# Label encoding
"""
le = LabelEncoder()
X[:,3]=le.fit_transform(X[:,3])
ohe=OneHotEncoder(categorical_features[3])
X = ohe.fit_transform(X).toarray()
"""

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#importing the  model

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

#fitting the model

lr.fit(X_train, y_train)

#Predict
y_pred=lr.predict(X_test)


print("predicted price ")
print(lr.predict([[2.7,2,15.5]]))

# Accuracy test

print('Accuracy Score :',lr.score(X,y))

# saving as a model usig pickle
"""
import pickle
with open('model_pickle','wb') as f:
      pickle.dump(lr,f)
"""

# saving the model using joblib
"""
import joblib
joblib.dump(lr,'model_joblib')
"""


