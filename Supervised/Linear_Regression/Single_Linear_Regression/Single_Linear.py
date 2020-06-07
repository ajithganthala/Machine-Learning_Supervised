# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

df= pd.read_csv('price.csv')
#print(df)

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color="red",marker='*')
plt.show()


#X = df['area'].values.reshape(-1,1)
#y = df['price'].values.reshape(-1,1)
X= df.iloc[:,0:1].values
y = df.iloc[:,1:].values
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
a=pd.Series([2500,1475,2220])
b=a.values.reshape(-1,1)
print(b)
print("predicted price for 2500")
print(lr.predict(b))


#printing Values

print('Vaue of Intercept :',lr.intercept_)
print('Value of Coefficient :',lr.coef_)
print('Accuracy Score :',lr.score(X,y))

#model evaluation

from sklearn import metrics
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))

print("RMSE Score :",rmse)


#comparision of acual values and predicted values

dt = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
print(dt)

#plotting the predicted ans actual values

df1 = dt.head(10)
df1.plot(kind='bar',figsize=(10,4))
plt.grid(which='major', linestyle='-', linewidth='0.2', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.2', color='black')
plt.show()


