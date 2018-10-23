#CGML Midterm - "Understanding intermediate layers using linear classifier probes"
#https://arxiv.org/pdf/1610.01644.pdf
#Camille Chow and Jacob Maarek
#Fall 2018
#Figure 3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

NUM_SAMP = 1000
training_size = 900
test_size = 100

#dataset: two guassian distributions
class Data():
  def __init__(self):
    np.random.seed()
    self.mu1 = np.random.randn()
    self.mu2 = np.random.randn()
    self.sig = .5
    self.x = np.atleast_2d(np.linspace(np.min([self.mu1, self.mu2])-2*self.sig, np.max([self.mu1, self.mu2])+2*self.sig,NUM_SAMP)).T
    self.g1 = np.atleast_2d(np.exp(-np.power(self.x - self.mu1, 2.) / (2 * np.power(self.sig, 2.))))
    self.g2 = np.atleast_2d(np.exp(-np.power(self.x - self.mu2, 2.) / (2 * np.power(self.sig, 2.))))
    
    self.coords1 = np.hstack((self.x, self.g1))
    self.coords2 = np.hstack((self.x, self.g2))
    
    self.data = np.vstack((self.coords1, self.coords2))
    self.labels = np.atleast_2d(np.hstack((np.zeros(NUM_SAMP), np.ones(NUM_SAMP)))).T
  
  def get_data(self):
    index = np.arange(2 * NUM_SAMP)
    choices1 = np.random.choice(index, size=training_size)
    choices2 = np.random.choice(index, size=test_size)
    return self.data[choices1], self.labels[choices1], self.data[choices2], self.labels[choices2]

num_layers = 33
layers = []
probes = []
losses = []
optims = []
varlist = []
threshold = .35
fail_count = 0
max_fails = 10

tf.reset_default_graph()

points = tf.placeholder(tf.float32, [None, 2])
choice = tf.placeholder(tf.float32, [None, 1])

def my_leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=.5)

#model: 32 dense layers w/ 128 hidden units, probes at every layer
for i in range (num_layers):
  if i == 0:
    layers.append(tf.layers.dense(points, 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), trainable=False))
       
  else:
    layers.append(tf.layers.dense(layers[i-1], 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), trainable=False))
    
  probes.append(tf.layers.dense(layers[i], 1, activation=None, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Probe"+str(i))))
  
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ("Probe" + str(i))))
  
  losses.append(tf.losses.sigmoid_cross_entropy(choice, probes[i]))
  optims.append(tf.train.RMSPropOptimizer(learning_rate = 0.1, momentum=0.75).minimize(losses[i], var_list=varlist[i]))

init = tf.global_variables_initializer()

probe_error_experiment = []

experiment_number = 0

#perform 100 experiments
while experiment_number < 100:
  sess = tf.Session()
  sess.run(init)
  
  print("Experiment", experiment_number)
  data = Data()
  x_train, y_train, x_test, y_test = data.get_data()
  probe_error = []
  
  layers_number = 0
  while layers_number < num_layers-1:
    
    for _  in range(0, 100):
      loss_np, _ = sess.run([losses[layers_number], optims[layers_number]], feed_dict={points: x_train, choice: y_train})
    #if probe doesn't train sufficiently, start over with new data
    if layers_number == 0 and loss_np > threshold:
      fail_count += 1
      if fail_count >= max_fails:
          print("Dataset Failed")
          layers_number = 0
          break
    else:
      probe_results, choice_vals = (sess.run([tf.round(tf.nn.sigmoid(probes[layers_number])), choice], feed_dict={points: x_test, choice: y_test}))
      probe_error.append (np.sum(np.round(np.abs(probe_results-choice_vals)))/(100))
      print("Layer", layers_number, "Loss", loss_np)
      layers_number += 1
    
  if fail_count >= max_fails:
    fail_count = 0
    sess.close()
    continue
  experiment_number += 1
  print(probe_error)
  probe_error_experiment.append(probe_error) 
  sess.close()

#take average of experiments
probe_error_experiment = np.asarray(probe_error_experiment)
average_layer = np.mean(probe_error_experiment, axis=0)
#plot
x = np.arange(1,num_layers)
plt.bar(x, average_layer, tick_label=x)
plt.xlabel("linear probe at layer k")
plt.ylabel("optimal prediction error")