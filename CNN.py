

# importing the required libraries and modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

    
#reading the training and the testing images data
data = pd.read_csv('D://Spring 2018/CS 559 Machine Learning/Project/train.txt', header = None)
testdata = pd.read_csv('D://Spring 2018/CS 559 Machine Learning/Project/test.txt', header = None)

X_train=data.iloc[:,0:64]
y_train=data.iloc[:,64]
X_test=testdata.iloc[:,0:64]
y_test=testdata.iloc[:,64]    


#a function to generate one hot encoding values of 10 d vector where each place represents 0 or 1 depending on the digit.
def one_hot_encode(vec,vals=10):
    n=len(vec)
    out=np.zeros((n,vals))
    out[range(n),vec]=1
    return out



class CNN():
    
    #initialize the values
    def __init__(self):
        self.i=0
        
        self.training_images=None
        self.training_labels=None
        self.testing_images=None
        self.testing_labels=None
    
    #setting up the training and testing images. Preprocessing and normalizing the data
    def set_up_images(self):
        print("Setting up training Images and labels")
        self.training_images=np.vstack([X_train])
        train_len= len(self.training_images)
        self.training_images=self.training_images.reshape(train_len,8,8,1)/16.0
        self.training_labels = one_hot_encode(np.hstack([y_train]),10)
        self.testing_images=np.vstack([X_test])
        test_len = len(self.testing_images)
        self.testing_images=self.testing_images.reshape(test_len,8,8,1)/16.0
        self.testing_labels=one_hot_encode(np.hstack([y_test]),10)
        return self.training_images,self.training_labels,self.testing_images,self.testing_labels

    #function created to feed the images in size of batches to the model.
    def next_batch(self,batch_size):
        x=self.training_images[self.i:self.i+batch_size]
        y=self.training_labels[self.i:self.i+batch_size]
        self.i=(self.i+batch_size) % len(self.training_images)
        return x,y

ch=CNN()
ch.set_up_images()

# initializing the weights as a random normal distribution
def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)

# defining the biases as 0.1 constant
def init_bias(shape):
    bias= tf.constant(0.1,shape=shape)
    return tf.Variable(bias)


#A convolution function that takes place in the Convolutional layer to extract the features
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

#After convolution, this function is being called for pooling
def max_pool_2by2(conv):
    return tf.nn.max_pool(conv,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

# A function Convolutional layer which has activation function as Rectified Linear unit
def convolutional_layer(x,shape):
    W=init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(conv2d(x,W) + b)

# Finally a fully layer densely connected neural network is formed 
def full_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,W) + b
    

#input image and the digits/values
X=tf.placeholder(tf.float32,shape=[None,8,8,1])
y_true=tf.placeholder(tf.float32,shape=[None,10])
#X_image = tf.reshape(X,[-1,8,8,1])

#2 convolutional layer followed by 2 pooling layers are used 
convo_1 = convolutional_layer(X,shape=[3,3,1,64])
convo_1_pool = max_pool_2by2(convo_1)

convo_2 = convolutional_layer(convo_1_pool,shape=[3,3,64,64])
convo_2_pool = max_pool_2by2(convo_2)

convo_2_flat=tf.reshape(convo_2_pool,[-1,2*2*64])
full_layer_one=tf.nn.relu(full_layer(convo_2_flat,1024))

#The predicted value from the fully connected layer i.e. 0 to 9
y_pred=full_layer(full_layer_one,10)

#cross entropyh loss function is being calculated using ypred and ytrue values
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))

#Adam Optimizer is used for minimizing the loss
optimizer=tf.train.AdamOptimizer(learning_rate=0.001)

train=optimizer.minimize(cross_entropy)


#initializing the global variables before running a tensorflow session
init=tf.global_variables_initializer()
steps=10000

#10000 steps are used and a batch size of 100 images are used in each step
with tf.Session() as sess:
    sess.run(init)
    #batch=ch.next_batch(100)
    for j in range(steps):
        
        batch=ch.next_batch(100)
        sess.run(train,feed_dict={X:batch[0],y_true:batch[1]})
        
        if(j%100==0):
            print("On Step: {}".format(j))
            print("Accuracy: ")
            
            matches=tf.equal(tf.argmax(y_pred,1),tf.argmax(y_true,1))
            acc=tf.reduce_mean(tf.cast(matches,tf.float32))
            print(sess.run(acc,feed_dict={X:testing_images,y_true:testing_labels}))
            print("\n")

