import pandas as pd
import numpy as np
from sklearn import metrics
dataset = pd.read_csv('petrol_consumption.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
print(dataset.mean()['Petrol_Consumption'])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestRegressor
rr = RandomForestRegressor(n_estimators=20, random_state=0)
rr.fit(X_train, y_train)
y_pred = rr.predict(X_test)
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
"""
With 20 trees, the root mean squared error is 64.93 which is greater
than 10 percent of the average petrol consumption i.e. 576.77.
This may indicate, among other things, that we have not used enough estimators (trees)
"""
