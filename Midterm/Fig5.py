#CGML Midterm - Linear Probes
#Figure 5

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

SEED = 66478
IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10

NUM_EPOCHS = 10
BATCH_SIZE = 32

def get_batch(x_train, y_train):
  choices = np.random.choice(np.arange(len(x_train)), size=BATCH_SIZE)
  return x_train[choices], y_train[choices]

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#shape test data
x_test = x_test.reshape(x_test.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
x_test = x_test.astype('float32')
x_test /= 255.0
y_test = tf.keras.utils.to_categorical(y_test, NUM_LABELS)

#shape training data
x_train = x_train.reshape(x_train.shape[0], IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS)
x_train = x_train.astype('float32')
x_train /= 255.0
y_train = tf.keras.utils.to_categorical(y_train, NUM_LABELS)

DATAPOINTS = len(x_train)
probes = []
probe_names = ["input", "conv1_preact", "conv1_postact", "conv1_postpool", "conv2_preact", "conv2_postact", "conv2_postpool", "fc1_preact", "fc1_postact", "logits"]
varlist = []
model_varlist = []
losses = []
optims = []

tf.reset_default_graph()

data = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
labels = tf.placeholder(tf.int32, [None, NUM_LABELS])
eval_data = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

def data_type():
  return tf.float32

def add_probe(input_layer, probe_num):  
  probes.append(tf.layers.flatten(input_layer))
  probes[probe_num] = tf.layers.dense(probes[probe_num], NUM_LABELS, activation=None, name=probe_names[probe_num], kernel_initializer=tf.glorot_normal_initializer(seed=None, dtype=tf.float32))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[probe_num]))
  losses.append(tf.losses.softmax_cross_entropy(labels, probes[probe_num]))
  optims.append(tf.train.RMSPropOptimizer(learning_rate=0.0005, decay=0.9, momentum=0.9, epsilon=1e-6, centered = True).minimize(losses[probe_num], var_list=varlist[probe_num]))
  
#model taken from: https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

# The variables below hold all the trainable weights. They are passed an
# initial value which will be assigned when we call:
# {tf.global_variables_initializer().run()}
conv1_weights = tf.Variable(tf.truncated_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=data_type()))
conv1_biases = tf.Variable(tf.zeros([32], dtype=data_type()))
conv2_weights = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=data_type()))
conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=data_type()))
fc1_weights = tf.Variable(tf.truncated_normal([IMAGE_SIZE // 4 * IMAGE_SIZE // 4 * 64, 512], stddev=0.1, seed=SEED, dtype=data_type()))
fc1_biases = tf.Variable(tf.constant(0.1, shape=[512], dtype=data_type()))
fc2_weights = tf.Variable(tf.truncated_normal([512, NUM_LABELS], stddev=0.1, seed=SEED, dtype=data_type()))
fc2_biases = tf.Variable(tf.constant(0.1, shape=[NUM_LABELS], dtype=data_type()))

# We will replicate the model structure for the training subgraph, as well
# as the evaluation subgraphs, while sharing the trainable parameters.
def model(data, train=False):
  add_probe(data, 0)
  conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME', name="conv1")
  add_probe(conv1, 1)  
  relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases), name="relu1")
  add_probe(relu1, 2)
  pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")
  add_probe(pool1, 3)
  conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME', name="conv2")
  add_probe(conv2, 4)
  relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases), name="relu2")
  add_probe(relu2, 5)
  pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")
  add_probe(pool2, 6)
  
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  reshape = tf.layers.flatten(pool2)

  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  fc1 = tf.matmul(reshape, fc1_weights, name="fc1") + fc1_biases
  add_probe(fc1, 7)
  hidden = tf.nn.relu(fc1, name="hidden")
  
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  
  add_probe(hidden, 8)
  
  return tf.matmul(hidden, fc2_weights) + fc2_biases

# Training computation: logits + cross-entropy loss.
logits = model(data, True)
add_probe(logits, 9)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# L2 regularization for the fully connected parameters.
regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
                tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# Add the regularization term to the loss.
loss += 5e-4 * regularizers

# Optimizer: set up a variable that's incremented once per batch and
# controls the learning rate decay.
batch = tf.Variable(0, dtype=data_type())
# Decay once per epoch, using an exponential schedule starting at 0.01.
learning_rate = tf.train.exponential_decay(
    0.01,                # Base learning rate.
    batch * BATCH_SIZE,  # Current index into the dataset.
    DATAPOINTS,          # Decay step.
    0.95,                # Decay rate.
    staircase=True)
# Use simple momentum for the optimization.
optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch) #, var_list=model_varlist???

def get_probe_error():
  probe_errors = []
  
  for a in range(10):
    loss_np = []
    optim_np = []
    correct = []
    
    print("training probe: ", a)
    for _ in (range(int(DATAPOINTS/BATCH_SIZE))):
      x_batch, y_batch = get_batch(x_train, y_train)
      loss_np, optim_np = sess.run([losses[a], optims[a]], feed_dict={data: x_batch, labels: y_batch})
      

    SET = int(DATAPOINTS/10)
    print("evaluating probe: ", a)
    for j in range (10):
      correct.append(sess.run([tf.nn.in_top_k(tf.nn.softmax(probes[a]), tf.argmax(labels, 1), 1)], feed_dict={data: x_train[int(j*SET):int((j+1)*SET-1), :], labels: y_train[int(j*SET):int((j+1)*SET-1), :]}))
  
    probe_errors.append((len(y_train)-np.sum(correct))/len(y_train))
    print(probe_names[a], "Probe_error:", (len(y_train)-np.sum(correct))/len(y_train))
  sess.close()
  return probe_errors

probe_errors = get_probe_error()

#plot accuracy at each probe - PRE TRAINING
plt.figure(figsize=(20,10))
index = range(len(probe_names))
plt.plot(index, probe_errors)
plt.xticks(index, probe_names)
plt.ylabel("test prediction error")

#train model
NUM_BATCHES = DATAPOINTS*NUM_EPOCHS//BATCH_SIZE
for i in range(NUM_BATCHES):
  x_batch, y_batch = get_batch(x_train, y_train)                                               
  loss_np_layers, optim_np_layers = sess.run([loss, optimizer], feed_dict={data: x_batch, labels: y_batch})
  
probe_errors_trained = get_probe_error()

# plot accuracy at each probe - POST TRAINING
plt.figure(figsize=(20,10))
index = range(len(probe_names))
plt.plot(index, probe_errors_trained)
plt.xticks(index, probe_names)
plt.ylabel("test prediction error")