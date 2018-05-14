
# importing the required libraries and modules
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm, metrics
from sklearn import decomposition
import time 


#reading the training and the testing images data
data = pd.read_csv('D://Spring 2018/CS 559 Machine Learning/Project/train.txt', header = None)
testdata = pd.read_csv('D://Spring 2018/CS 559 Machine Learning/Project/test.txt', header = None)


#normalizing the training data 
X_train=data.iloc[:,0:64]/16.0
y_train=data.iloc[:,64]
X_test=testdata.iloc[:,0:64]/16.0
y_test=testdata.iloc[:,64]


######################################################
######################################################
######################################################
#SVM without PCA


parameters = {'kernel':'rbf', 'C':1, 'gamma': 1,'degree':3}
classifier = svm.SVC(kernel=parameters['kernel'],gamma=parameters['gamma'],C = parameters['C'],degree=parameters['degree'])
classifier.fit(X_train,y_train)


predicted = classifier.predict(X_test)
count=0
for i,j in zip(predicted,y_test):
    if (i == j):
        count+=1
print("\n")
print("Accuracy using SVM without using PCA is: ",count/len(y_test) )        




######################################################
######################################################
######################################################
# Applying PCA 
#finding the important principalcomponents using the variance explained by them
pca=PCA()
pca.fit(X_train)


with plt.style.context('fivethirtyeight'):    
    plt.show()
    plt.xlabel("Principal components ")
    plt.ylabel("Variance")
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Variance Explained by Extracted Componenent')

plt.show()
time.sleep(5)

 
######################################################
######################################################
######################################################
#PCA with 15 components

pca=PCA(n_components=15)

train_ext=pca.fit_transform(X_train)

#Gaussian radial basis kernel is used
parameters = {'kernel':'rbf', 'C':1, 'gamma': 1,'degree':3}
classifier = svm.SVC(kernel=parameters['kernel'],gamma=parameters['gamma'],C = parameters['C'],degree=parameters['degree'])
classifier.fit(train_ext,y_train)

test_ext=pca.transform(X_test)


predicted = classifier.predict(test_ext)
count=0
for i,j in zip(predicted,y_test):
    if (i == j):
        count+=1
print("\n")
print("Accuracy using SVM with PCA 15 components is: ",count/len(y_test) )        




######################################################
######################################################
######################################################
#PCA with 20 components

pca=PCA(n_components=20)
pca.fit(X_train)

train_ext=pca.fit_transform(X_train)

parameters = {'kernel':'rbf', 'C':1, 'gamma': 1,'degree':3}
classifier = svm.SVC(kernel=parameters['kernel'],gamma=parameters['gamma'],C = parameters['C'],degree=parameters['degree'])
classifier.fit(train_ext,y_train)

test_ext=pca.transform(X_test)

predicted = classifier.predict(test_ext)
count=0
for i,j in zip(predicted,y_test):
    if (i == j):
        count+=1
print("\n")
print("Accuracy using SVM with PCA 20 components is: ",count/len(y_test) )        




######################################################
######################################################
######################################################
#PCA with 25 components

pca=PCA(n_components=25)
pca.fit(X_train)

train_ext=pca.fit_transform(X_train)

parameters = {'kernel':'rbf', 'C':1, 'gamma': 1,'degree':3}
classifier = svm.SVC(kernel=parameters['kernel'],gamma=parameters['gamma'],C = parameters['C'],degree=parameters['degree'])
classifier.fit(train_ext,y_train)

test_ext=pca.transform(X_test)

predicted = classifier.predict(test_ext)
count=0
for i,j in zip(predicted,y_test):
    if (i == j):
        count+=1
print("\n")
print("Accuracy using SVM with PCA 25 components is: ",count/len(y_test) )        


#Hence, found SVM with PCA of 20 components is a good result with 97.77 % Accuracy 