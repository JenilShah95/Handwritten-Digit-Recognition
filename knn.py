
#importing the required libraries
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier


#reading the training and the testing images data    
data = pd.read_csv('D://Spring 2018/CS 559 Machine Learning/Project/train.txt', header = None)
testdata = pd.read_csv('D://Spring 2018/CS 559 Machine Learning/Project/test.txt', header = None)


#normalizing the training data 
X_train=data.iloc[:,0:64]/16.0
y_train=data.iloc[:,64]
X_test=testdata.iloc[:,0:64]/16.0
y_test=testdata.iloc[:,64]



#After applying PCA, it was found 20 principal components are enough
pca = PCA(20)

train_ext=pca.fit_transform(X_train)
test_ext=pca.transform(X_test)


components = [15,20]
neighbors = [1,2,3,4,5]

scores = np.zeros( (components[len(components)-1]+1, neighbors[len(neighbors)-1]+1 ) )

#running the KNN for 15 and 20 components with k's value being 1 to 5
for component in components:
    for n in neighbors:
        knn = KNeighborsClassifier(n_neighbors=n)
        knn.fit(train_ext[:,:component], y_train)
        score = knn.score(test_ext[:,:component], y_test)
        #predict = knn.predict(X_test_pca[:,:component])
        scores[component][n] = score
        
        print('Components = ', component, ', neighbors = ', n,', Score = ', score)

