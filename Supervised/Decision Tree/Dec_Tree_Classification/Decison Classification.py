import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree

dt = pd.read_csv('weather.csv')
print(dt.head())
dt['play'],play_conform=pd.factorize(dt['play'])
dt['outlook'],_=pd.factorize(dt['outlook'])
dt['temperature'],_=pd.factorize(dt['temperature'])
dt['humidity'],_=pd.factorize(dt['humidity'])
dt['windy'],_=pd.factorize(dt['windy'])
print(dt.head())

X = dt.iloc[:,:-1]
y = dt.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=0)

df = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3, random_state=0)
df.fit(X_train, y_train)

y_pred = df.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cm)

a = cm.shape
crctpred = 0
falsepred = 0

for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            crctpred +=cm[row,c]
        else:
            falsepred += cm[row,c]

print('Correct predictions: ', crctpred)
print('False predictions', falsepred)
#print(y_pred)
#print('Accuracy',metrics.accuracy_score(y_test, y_pred))

