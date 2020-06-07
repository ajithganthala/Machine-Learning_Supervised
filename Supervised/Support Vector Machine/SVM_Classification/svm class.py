import pandas as pd
df=pd.read_csv('winequality-red.csv')

print(df.head())

print(df.isna().sum())

from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
new_data=scaler.fit_transform(df.drop(labels=['quality'],axis=1))

print(df.columns)

columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']

new_df=pd.DataFrame(data=new_data,columns=columns)

print(new_df.head())

x=new_df
y=df['quality']

from  sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.33, random_state=42)

from sklearn.svm import SVC

model=SVC(kernel='rbf',C=0.1)
model.fit(train_x,train_y)

model.predict(test_x)

from sklearn.metrics import accuracy_score
print(accuracy_score(test_y,model.predict(test_x)))


# As observed, the accuracy of the model is quite low. We need to implement the grid search approach to optimize the parameters to give the best accuracy.

# Implementing Grid Search

from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1,1,10,50,100,500],'gamma':[1,0.5,0.1,0.01,0.001]}

grid= GridSearchCV(SVC(),param_grid, verbose=3, n_jobs=-1)

grid.fit(train_x,train_y)

print(grid.best_params_)

model_new=SVC(C=10, gamma=1)
model_new.fit(train_x,train_y)

print(accuracy_score(test_y,model_new.predict(test_x)))

