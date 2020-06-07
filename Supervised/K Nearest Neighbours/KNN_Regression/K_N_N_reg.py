import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import seaborn as sns
from math import sqrt
from sklearn.metrics import mean_squared_error 
df=pd.read_csv('train.csv')
X = df[['Height(Inches)']]
y = df[['Weight(Pounds)']]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
knn = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
error = sqrt(mean_squared_error(y_test,y_pred))
print(error)
from sklearn import metrics
rmse_val = []
dic = {}
#To find the exact K value
for K in range(20):
    
    model =KNeighborsRegressor(n_neighbors = K+1, metric='euclidean')
    model.fit(X_train, y_train)  #fit the model
    pred=model.predict(X_test) #make prediction on test set
    error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
    rmse_val.append(error) #store rmse values
    
    dic[rmse_val[K]] = K+1
    K_sorted=sorted(rmse_val,key=float)
    val = K_sorted[0]
print("the best k values is: ",val,"and rmse is: ",dic[val])
print(metrics.r2_score(y_test,y_pred))

curve = pd.DataFrame(rmse_val) #elbow curve 
curve.plot()
plt.show()

