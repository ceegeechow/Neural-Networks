#CGML Midterm - Linear Probes
#Figure 8

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import backend as K
import keras

#Convert train data into val and train
num_classes = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train/255
x_train = np.reshape(x_train, (len(x_train), 28*28))
x_test = x_test/255

DATAPOINTS = len(x_train)
NUM_MINIBATCHES = 5000
NUM_EPOCHS = 10
BATCH_SIZE = int(DATAPOINTS*NUM_EPOCHS/NUM_MINIBATCHES)

def get_batch(x_train, y_train):
  choices = np.random.choice(np.arange(len(x_train)), size=BATCH_SIZE)
  return x_train[choices], y_train[choices]

#model: 128 fully connected layers, skip connection from input to layer 64
num_layers = 128
layers = []
probes = []
probe_losses = []
probe_optims = []
varlist = []
layers_losses = []
linear_classifier = []

tf.reset_default_graph()

images = tf.placeholder(tf.float32, [None, 28*28])
image_labels = tf.placeholder(tf.float32, [None, 10])

def my_leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=.5)

for i in range (num_layers):
  if i == 0:
    layers.append(tf.layers.dense(images, 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Layer"+str(i))))
       
  elif i == 63:
    layers.append(tf.layers.dense(layers[i-1], 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Layer_1_"+str(i))))
    layers[i] = layers[i] + tf.layers.dense(images, 128, activation=None, use_bias = False, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Layer_2_"+str(i)))
    
  else:
    layers.append(tf.layers.dense(layers[i-1], 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Layer"+str(i))))  
                         
  probes.append(tf.layers.dense(layers[i], 10, activation=None, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Probe"+str(i))))

  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ("Probe" + str(i))))

  probe_losses.append(tf.losses.softmax_cross_entropy(image_labels, probes[i]))
  probe_optims.append(tf.train.RMSPropOptimizer(learning_rate=0.0005, decay=0.9, momentum=0.9, epsilon=1e-6, centered = True).minimize(probe_losses[i], var_list=varlist[i]))

linear_classifier = tf.layers.dense(layers[i], 10, activation=None, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Layer"+str(i+1)))
layer_loss = tf.losses.softmax_cross_entropy(image_labels, linear_classifier)

layer_varlist = list(filter(lambda a : "Layer" in a.name, [v for v in tf.trainable_variables()]))
layer_optim = tf.train.RMSPropOptimizer(learning_rate = 0.00001, momentum=0.9, centered = True).minimize(layer_loss, var_list = layer_varlist)
    
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

probe_error_batch_num = []    

for q in range(NUM_MINIBATCHES):
  x_batch, y_batch = get_batch(x_train, y_train)                                                
  loss_np_layer, optim_np_layer = sess.run([layer_loss, layer_optim], feed_dict={images: x_batch, image_labels: y_batch})
  
  if q%100 == 0:
    print("MiniBatch:", q, "MiniBatch Loss: ", loss_np_layer)  
                         
  if q == 0 or q == 499 or q == 4999:
    print("calculate probes")
    probe_error = []

    for a in range(num_layers):      
      loss_np = []
      optim_np = []
      
      for t in range (int(DATAPOINTS/BATCH_SIZE)):
        x_batch, y_batch = get_batch(x_train, y_train)
        loss_np, optim_np = sess.run([probe_losses[a], probe_optims[a]], feed_dict={images: x_batch, image_labels: y_batch})

      correct = sess.run([tf.nn.in_top_k(tf.nn.softmax(probes[a]), tf.argmax(image_labels, 1), 1)], feed_dict={images: x_train, image_labels: y_train})

      probe_error.append((len(y_train)-np.sum(correct))/len(y_train))
      print("Probe:", a+1, "Probe_error:", (len(y_train)-np.sum(correct))/len(y_train))
      
    print(probe_error)  
    probe_error_batch_num.append(probe_error) 

x = np.arange(1,num_layers+1)
plt.figure(figsize=(20,10))
plt.bar(x, probe_error_batch_num[0])
plt.xlabel("linear probe at layer k")
plt.ylabel("optimal prediction error")
axes = plt.gca()
axes.set_ylim([0.0,1.0])

plt.figure(figsize=(20,10))
plt.bar(x, probe_error_batch_num[1])
plt.xlabel("linear probe at layer k")
plt.ylabel("optimal prediction error")
axes = plt.gca()
axes.set_ylim([0.0,1.0])

plt.figure(figsize=(20,10))
plt.bar(x, probe_error_batch_num[2])
plt.xlabel("linear probe at layer k")
plt.ylabel("optimal prediction error")
axes = plt.gca()
axes.set_ylim([0.0,1.0])