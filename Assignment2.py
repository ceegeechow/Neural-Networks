#Camille Chow
#ECE 471 Assignment 2

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm

BATCH_SIZE = 40
NUM_BATCHES = 300

class Data(object):
    def __init__(self):
        num_samp = 300
        sigma = 0.1
        np.random.seed(1)           #change after testing

        self.index = np.arange(num_samp)        
        self.t1 = np.random.uniform(.08, 1.8, num_samp)            #need adjustment?
        self.x1 = 6 * self.t1 * np.cos(2*np.pi*self.t1) + np.random.normal(0, sigma, num_samp)
        self.y1 = 6 * self.t1 * np.sin(2*np.pi*self.t1) + np.random.normal(0, sigma, num_samp)

        self.t2 = np.random.uniform(.08, 1.8, num_samp)            #need adjustment?
        self.x2 = 6 * self.t2 * np.cos(2*np.pi*self.t2 + np.pi) + np.random.normal(0, sigma, num_samp)
        self.y2 = 6 * self.t2 * np.sin(2*np.pi*self.t2 + np.pi) + np.random.normal(0, sigma, num_samp)
        
    def get_batch(self):
        choices = np.random.choice(self.index, size=BATCH_SIZE)

        return self.x[choices], self.y[choices].flatten()

# def f(x):
# 	#to be optimized:
#     w = tf.get_variable('w', [M, 1], tf.float32, tf.random_normal_initializer())    
#     b = tf.get_variable('b', [], tf.float32, tf.zeros_initializer())
#     mu = tf.get_variable('mu', [M, 1], tf.float32, tf.random_uniform_initializer())    
#     sig = tf.get_variable('sig', [M, 1], tf.float32, tf.random_uniform_initializer())
#     #calculate phi
#     phi = tf.exp(- tf.pow((x - mu)/sig,2))                                          #dimension mismatch????
#     #calculate yhat
#     return tf.squeeze(tf.matmul(tf.transpose(w),phi) + b)

# x = tf.placeholder(tf.float32, [BATCH_SIZE])
# y = tf.placeholder(tf.float32, [BATCH_SIZE])
# y_hat = f(x)

# loss = .5 * tf.reduce_mean(tf.pow(y_hat - y, 2))
# optim = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)
# init = tf.global_variables_initializer()

# sess = tf.Session()
# sess.run(init)

# data = Data()
# #perform gradient descent
# for _ in tqdm(range(0, NUM_BATCHES)):
#     x_np, y_np = data.get_batch()
#     loss_np, _ = sess.run([loss, optim], feed_dict={x: x_np, y: y_np})
#plots
plt.figure(1, figsize=[18,12])

data = Data()
plt.scatter(-data.x1,data.y1)
plt.scatter(-data.x2,data.y2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('Spirals')

plt.show()