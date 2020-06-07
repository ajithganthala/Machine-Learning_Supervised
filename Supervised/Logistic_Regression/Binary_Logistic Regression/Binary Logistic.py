import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#loading data set
df = pd.read_csv("mag.csv")
#print(df.head())

#checking for null va;ues
#print(df.isnull().sum())

"""#exploring target variable and vsualising it
df.Buy.value_counts()
sns.countplot(x = 'Buy', data = df, palette = 'hls')
plt.show()"""

#takking data to X and Y
X = df.iloc[:,1:-1]

y = df['Buy']

# Splitting the dataset into the Training set and Test set

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=2)

# Fitting Multiclass Logistic Classification to the Training set

lr = LogisticRegression()
lr.fit(X_train,y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)
print("Predicted VAlues:",y_pred)

# Making the Confusion Matrix

cm = metrics.confusion_matrix(y_test, y_pred)
print("Confusion MAtrix:",cm)

#Visualisation of CNF MAtrix

class_names=[1,2]
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
sns.heatmap(pd.DataFrame(cm), annot=True, cmap="viridis" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

#Acuuracy of the model
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
"""print('Precision:', 0.6923076923076923)
print('Recall:', 0.8571428571428571)"""

"""#ROC CUrve
y_pred_proba = lr.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()"""
