import pandas as pd

df = pd.read_csv('iris.csv')
print(df.head())

# **Train Using Support Vector Machine (SVM)**

from sklearn.model_selection import train_test_split


X = df.iloc[:,0:-1]
y = df.iloc[-1]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4,random_state=0)


print(len(X_train))


print(len(X_test))


from sklearn.svm import SVC
model = SVC()


model.fit(X_train, y_train)


print(model.score(X_test, y_test))


print(model.predict([[4.8,3.0,1.5,0.3]]))


# **Tune parameters**

# **1. Regularization (C)**

model_C = SVC(C=1)
model_C.fit(X_train, y_train)
print(model_C.score(X_test, y_test))

model_C = SVC(C=10)
model_C.fit(X_train, y_train)
print(model_C.score(X_test, y_test))


# **2. Gamma**
model_g = SVC(gamma=10)
model_g.fit(X_train, y_train)
print(model_g.score(X_test, y_test))

# **3. Kernel**
model_linear_kernal = SVC(kernel='linear')
model_linear_kernal.fit(X_train, y_train)
print(model_linear_kernal.score(X_test, y_test))

