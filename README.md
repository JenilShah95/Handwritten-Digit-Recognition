# Handwritten-Digit-Recognition

I used the dataset provided on the UCI Machine Learning Repository (
https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits).
The dataset consists of approximately 4000 training images and 1600 testing images representing a digit. Each input image is being represented by 64 features which can be reshaped as 8*8 pixels to represent an image. Below I have shown an example of an input image generated from the data.

# Algortihms Used

# Convolutional Neural Networks

For my project, I implemented CNN using the Tensorflow module in Python and have used CNN as follows. First, I have passed the input image as an input to the neural nets along with a randomly distributed weights and biases. The input image is passed on to the Convolutional layer where the feature extraction takes place and the extracted features are passed to the rectified linear unit i.e. the activation function to convert the negative values to 0. The transformed feature is then being pooled by a Pooling layer, where I have used a max Pool of 2*2 matrix. The same process takes place at the second convolutional layer to extract the features. Finally, the extracted 1024 features are passed to a fully connected dense neural network which finally gives the output as a value for each digit being the probability between 0 to 1.
The loss function used is cross entropy loss function and the model is being trained to optimize or reduce the loss function using the Gradient Descent Optimizer. CNN produced the highest accuracy of about 97.5% among the three algorithms.

# Support Vector Machines

Support Vector Machines is another classification algorithm that I have used. SVM helps build a hyperplane between the two classes using the data points referred to as the Support Vectors and the aim is to build a hyperplane such that it maximizes the distance between the support vectors from the hyperplane for both the classes.
I implemented the SVM using the SVC classifier of the scikit-learn library module. I have used non linear SVC for the classification problem. Various parameter values like C, gamma and kernel types were explored. C is defined as the error term i.e. it tells the SVM how much to avoid the misclassification or the error rate so as to optimize the result. Gamma is the free parameter which tells about the variance that is developed by the support vectors. Since, this is the non linear classification, I have used Gaussian radial basis function as the kernel of SVC. The technique of classification being used is “One vs all”.

# K Nearest Neighbors
The K nearest Neighbour algorithm is another classification technique, which classifies a test data point according to the data points that are present near that test point. The “k” represents the number of neighbouring points to look around the test data point while classifying the point.
I have used KNeighborsClassifier library of the scikit-learn module. For experimenting, I have used different values of k from 1 to 5 and also different components/input features foe every values of k which are reduced using PCA analysis.

# Principal Component Analysis

Principal Component Analysis is the dimensional reduction technique used for reducing the number of input features that help in increasing the accuracy of the model and reduce the running time of the model.
I applied SVM with PCA and KNN with PCA and the results were influenced greatly. Scikit-learn’s library PCA is used in Python to carry out the PCA and the method “explained_variance_” helps us in identifying the number of features that are required.
As we can see above that out of 64 features only 15 to 20 features are helpful in explaining the data (variance is a bit higher). Hence, I decided to test the system using the reduced features and found that the results were influenced greatly.


# Results

I. Convolutional Neural Nets
After training the input image for about 10000 steps and in a batch size of 100 images at a time, the testing accuracy was found a highly impressive value of nearly 98.5%.
The Adam Optimizer was used for backpropagation and minimization of the error function i.e. cross entropy loss function.

II. Support Vector Machines
I ran SVC library in python with a C value of 1 and gamma value of 1 and the kernel as rbf kernel as it suits best for the non linear data.
First, I ran the classifier on the original data (data was normalized to bring in the range 0-1). The output was obtained as shown below.
The accuracy was found to be really impressive but after applying PCA, and reducing the number of features to 20, the best results were obtained with an accuracy of about nearly 98%.

III. K Nearest Neighbours
After applying PCA, I found out previously that 15 to 20 components are enough to explain the variance and hence are enough for the input instead of 64 features.
Hence, after running the K nearest neighbour algorithm for 15 and 20 components and the neighbours values from 1 to 5, I found the following results.
We can see that 20 components well define the input data and provide a good accuracy result on the test data with the neighbours value of 4 and 5.
