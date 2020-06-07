# necessary Imports
import pandas as pd
import matplotlib.pyplot as plt


df= pd.read_csv('Admission_Prediction.csv')

print(df.head())

print(df.isna().sum())


# As we can see, there are some column with missing values. we need to impute those missing values.
df['GRE Score'].fillna(df['GRE Score'].mean(),inplace=True)
df['TOEFL Score'].fillna(df['TOEFL Score'].mean(),inplace=True)
df['University Rating'].fillna(df['University Rating'].mode()[0],inplace=True)

# seeing that after imputation no column has missing values
print(df.isna().sum())

x=df.drop(['Chance of Admit','Serial No.'],axis=1)
y=df['Chance of Admit']
columns=x.columns

from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33, random_state=33)

from sklearn.svm import SVR
svr= SVR(C=10)

svr.fit(train_x, train_y)


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
