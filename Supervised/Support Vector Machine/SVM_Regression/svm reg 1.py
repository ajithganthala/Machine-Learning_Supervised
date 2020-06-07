import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv('Position_Salaries.csv')

print(dataset.head())

x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2:].values

from sklearn.preprocessing import StandardScaler

st_x=StandardScaler()
st_y=StandardScaler()

X=st_x.fit_transform(x)
Y=st_y.fit_transform(y)

fig=plt.figure()
ax=fig.add_axes([0,0,1,1])
ax.scatter(X,Y,color='r')
plt.show()

from sklearn.svm import SVR


regressor=SVR(kernel='rbf')

regressor.fit(X,Y)


plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='blue')
from sklearn.metrics import r2_score
score= r2_score(test_y,svr.predict(test_x))
print(score)

from sklearn.model_selection import GridSearchCV
param_grid={'C':[0.1,1,10,50,100,500],'gamma':[1,0.5,0.1,0.01,0.001] }
grid= GridSearchCV(SVR(),param_grid, verbose=3)

grid.fit(train_x,train_y)
grid.best_estimator_


svr_new=SVR(C=50,degree=3, epsilon=0.1, gamma=0.001,kernel='rbf')

svr_new.fit(train_x, train_y)

score_new= r2_score(test_y,svr_new.predict(test_x))
print(score_new)
