import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import seaborn as sns

# Importing the dataset
df = pd.read_csv('iris.csv')
#print(df.head())

#IRisi Image SHow

"""img=mpimg.imread('iris_types.jpg')
plt.figure(figsize=(20,40))
plt.axis('off')
imgplot = plt.imshow(img)
plt.show()"""



#Spliting the dataset in independent and dependent variables
X = df.iloc[:,:4]
y = df['species']

#print(X)
#print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 2)

"""# Feature Scaling to bring the variable in a single scale
from sklearn.preprocessing import StandardScaler
sd = StandardScaler()
X_train = sd.fit_transform(X_train)
X_test = sd.transform(X_test)"""

# Fitting Multiclass Logistic Classification to the Training set
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lr.predict(X_test)
print("Predicted VAlues:",y_pred)

#lets see the actual and predicted value side by side
y_compare = np.vstack((y_test,y_pred)).T
#actual value on the left side and predicted value on the right hand side
#printing the top 5 values
y_compare[:5,:]


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:",cm)

#Visualisation of CNF MAtrix

class_names=[1,2,3]
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

#lets find out of no.of true and false predictions
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
print ('Accuracy of the multiclass logistic classification is: ', crctpred/(cm.sum()))


