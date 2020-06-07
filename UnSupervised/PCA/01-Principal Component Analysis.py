#IMporting Necessary Libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

#Loading Data Set from SKLEARN Library
from sklearn.datasets import load_breast_cancer

#Creating a variable to dataset
df = load_breast_cancer()
#print(df.head())

#As our inbuilt dataset will be in dictionary format will be seing the information by printing the keys of dictioanry
print(df.keys())

#we can see the description of the data set
print(df['DESCR'])

#Here will be coverting dictionary to pandas data frame.
dt = pd.DataFrame(df['data'],columns=df['feature_names'])

print(dt.head())

#PCA Visualization 
#As we've noticed before it is difficult to visualize high dimensional data, we can use PCA to find the first two principal components, and visualize the data in this new, two-dimensional space, with a single scatter-plot. Before we do this though, we'll need to scale our data so that each feature has a single unit variance.

#Importing Preprocessing Libraries
#This standard scalar is responsible for coverting the data of any distribution to standard normalised one
from sklearn.preprocessing import StandardScaler

#Creating and object to model
scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

#fitting the data
scaler.fit(dt)

#We are transforming our old data to new scaled data
data = scaler.transform(dt)

# PCA with Scikit Learn uses a very similar process to other preprocessing functions that
#come with SciKit Learn. We instantiate a PCA object, find the principal components
#using the fit method, then apply the rotation and dimensionality reduction by calling transform().
# We can also specify how many components we want to keep when creating the PCA object.


from sklearn.decomposition import PCA

pca = PCA(n_components=2) #n_components reprwsnts How many features we need to keep.

pca.fit(data)


# Now we can transform this data to its first 2 principal components.
x_pca = pca.transform(data)

# ShPE OF THE data before dimesionality reduction
print(data.shape)

#Shape of data after dimensionality reduction
print(x_pca.shape)


#We've reduced 30 dimensions to just 2! Let's plot these two dimensions out!
#Why we do that ? We do it get more accuracy of the model

plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],c=df['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
plt.show()


#Clearly by using these two components we can easily separate these two classes.
#Interpreting the components 
#Unfortunately, with this great power of dimensionality reduction, comes the cost of being able to easily understand what these components represent.
#The components correspond to combinations of the original features, the components themselves are stored as an attribute of the fitted PCA object:

print(pca.components_)


# In this numpy matrix array, each row represents a principal component, and each column relates back to the original features. we can visualize this relationship with a heatmap:

df_comp = pd.DataFrame(pca.components_,columns=df['feature_names'])

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma',)
plt.show()


#This heatmap and the color bar basically represent the correlation between the various feature and the principal component itself.
# 
