import tensorflow as tf
import numpy as np
"""
A simple numpy implementation of a XOR gate to understand the backpropagation
algorithm

"""

x=tf.placeholder(tf.float64,shape=[4,2],name="x")#declaring a place holder for input x
y=tf.placeholder(tf.float64,shape=[4,1],name="y")#declaring a place holder for desired output y

m=np.shape(x)[0]#number of training examples
n=np.shape(x)[1]#number of features
hidden_s = 2 #number of nodes in the hidden layer
l_r = 0.1#learning rate initialization

"""
We initialize the weights that our model learns first as matrices of random
variables  which are then updated by backpropagation. 

generally the dimensions for each weight vector is: (size of current layer+1,size of next layer)
"""
theta1=tf.cast(tf.Variable(tf.random_uniform([3,hidden_s]),name="theta1"),tf.float64)
theta2=tf.cast(tf.Variable(tf.random_uniform([hidden_s+1,1]),name="theta2"),tf.float64)

#conducting forward propagation

#first, a column of biases is added to the input of the first layer
a1=tf.concat([np.c_[np.ones(x.shape[0])],x],1)
#the weights of the first layer are multiplied by the input of the first layer
z1=tf.matmul(a1,theta1)
#the input of the second layer is the output of the first layer, passed through the activation function and column of biases is added
a2=tf.concat([np.c_[np.ones(x.shape[0])],tf.sigmoid(z1)],1)
#the input of the second layer is multiplied by the weights
z3=tf.matmul(a2,theta2)
#the output is passed through the activation function to obtain the final probability
h3=tf.sigmoid(z3)


"""
declaring the cost function that has to be minimized to obtain the weights.it is the 
negative sum of the product of y and log of the hypothesis and (1-y) and log of (1-hyp)
"""
cost_func=-tf.reduce_sum(y*tf.log(h3)+(1-y)*tf.log(1-h3),axis=1)

#built in tensorflow optimizer that conducts gradient descent using specified learning rate to obtain theta values
optimiser = tf.train.GradientDescentOptimizer(learning_rate=l_r).minimize(cost_func)


#setting required X and Y values
X=[[0,0],[0,1],[1,0],[1,1]]
Y=[[0],[1],[1],[0]]

#initializing all variables, creating a session and running a tensorflow session
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

#running gradient descent for each iteration and printing the hypothesis obtained using the updated theta values
for i in range(100000):
    sess.run(optimiser, feed_dict={x:X,y:Y})#setting place holder values using feed_dict
    if i%100==0:
        print("Epoch:",i)
        print("Hyp:",sess.run(h3,feed_dict={x:X,y:Y}))
