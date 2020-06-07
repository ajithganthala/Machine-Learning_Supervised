"""https://archive.ics.uci.edu/ml/datasets/banknote+authentication"""
import pandas as pd
import numpy as np
dataset = pd.read_csv('bill_authentication.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
rc = RandomForestClassifier(n_estimators=30, random_state=0)
rc.fit(X_train, y_train)
y_pred = rc.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
print("Confusion matrix:",confusion_matrix(y_test,y_pred))
print("Classification report:",classification_report(y_test,y_pred))
print("Accuracy Score:",accuracy_score(y_test, y_pred))




