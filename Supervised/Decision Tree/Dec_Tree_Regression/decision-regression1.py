# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics

df = pd.read_csv('train.csv')
#print(df.head())

x = df.iloc[:,0:2]
y = df.iloc[:,-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25,train_size=0.75, random_state=40)

dtree = DecisionTreeRegressor()
dtree.fit(x_train,y_train)
y_pred = dtree.predict(x_test)
print(y_pred)

rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print("RMSE Score :",rmse)
dt = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(dt)
