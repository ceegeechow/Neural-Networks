probes = []
probe_names = ["input", "conv1_preact", "conv1_postact", "conv1_postpool", "conv2_preact", "conv2_postact", "conv2_postpool", "fc1_preact", "fc1_postact", "logits"]
varlist = []
losses = []
optims = []
lr = .1
momentum = .75
epochs = 10
probe_errors = []

data = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
labels = tf.placeholder(tf.float32, [None, NUM_LABELS])

def data_type():
  return tf.float32

def add_probe(input_layer, probe_num):
  probes.append(tf.layers.dense(input_layer, NUM_LABELS, activation=None, name=probe_names[probe_num]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[probe_num]))
  losses.append(tf.losses.softmax_cross_entropy(labels, probes[probe_num]))
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[probe_num], var_list=varlist[probe_num]))
  
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
  probes.append(tf.layers.dense(data, NUM_LABELS, activation=None, name=probe_names[0]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[0]))
  losses.append(tf.losses.softmax_cross_entropy(labels, probes[0]))
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[0], var_list=varlist[0]))
  
  conv1 = tf.nn.conv2d(data, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
  
  probes.append(tf.layers.dense(conv1, NUM_LABELS, activation=None, name=probe_names[1]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[1]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[1], var_list=varlist[1]))
  
  relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

  probes.append(tf.layers.dense(relu1, NUM_LABELS, activation=None, name=probe_names[2]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[2]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[2], var_list=varlist[2]))
  
  pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  probes.append(tf.layers.dense(pool1, NUM_LABELS, activation=None, name=probe_names[3]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[3]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[3], var_list=varlist[3]))  
  
  conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
  
  probes.append(tf.layers.dense(conv2, NUM_LABELS, activation=None, name=probe_names[4]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[4]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[4], var_list=varlist[4]))  
  
  relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
  
  probes.append(tf.layers.dense(relu2, NUM_LABELS, activation=None, name=probe_names[5]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[5]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[5], var_list=varlist[5])) 
  
  pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
  
  probes.append(tf.layers.dense(pool2, NUM_LABELS, activation=None, name=probe_names[6]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[6]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[6], var_list=varlist[6])) 
  
  # Reshape the feature map cuboid into a 2D matrix to feed it to the
  # fully connected layers.
  pool_shape = pool.get_shape().as_list()
  reshape = tf.reshape(pool, [pool_shape[0], pool_shape[1] * pool_shape[2] * pool_shape[3]])
  # Fully connected layer. Note that the '+' operation automatically
  # broadcasts the biases.
  fc1 = (tf.matmul(reshape, fc1_weights) + fc1_biases)
  
  probes.append(tf.layers.dense(fc1, NUM_LABELS, activation=None, name=probe_names[7]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[7]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[7], var_list=varlist[7])) 
  
  fc1_postact = tf.nn.relu(fc1)
  
  probes.append(tf.layers.dense(fc1_postact, NUM_LABELS, activation=None, name=probe_names[8]))
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[8]))
  losses.append()
  optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[8], var_list=varlist[8]))
  
  # Add a 50% dropout during training only. Dropout also scales
  # activations such that no rescaling is needed at evaluation time.
  if train:
    hidden = tf.nn.dropout(hidden, 0.5, seed=SEED)
  return tf.matmul(hidden, fc2_weights) + fc2_biases

# Training computation: logits + cross-entropy loss.
logits = model(train_data_node, True)

probes.append(tf.layers.dense(logits, NUM_LABELS, activation=None, name=probe_names[9]))
varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, probe_names[9]))
losses.append()
optims.append(tf.train.RMSPropOptimizer(learning_rate=lr, momentum=momentum).minimize(losses[9], var_list=varlist[9]))



# loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

# # L2 regularization for the fully connected parameters.
# regularizers = (tf.nn.l2_loss(fc1_weights) + tf.nn.l2_loss(fc1_biases) +
#                 tf.nn.l2_loss(fc2_weights) + tf.nn.l2_loss(fc2_biases))
# # Add the regularization term to the loss.
# loss += 5e-4 * regularizers

# # Optimizer: set up a variable that's incremented once per batch and
# # controls the learning rate decay.
# batch = tf.Variable(0, dtype=data_type())
# # Decay once per epoch, using an exponential schedule starting at 0.01.
# learning_rate = tf.train.exponential_decay(
#     0.01,                # Base learning rate.
#     batch * BATCH_SIZE,  # Current index into the dataset.
#     train_size,          # Decay step.
#     0.95,                # Decay rate.
#     staircase=True)
# # Use simple momentum for the optimization.
# optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(loss, global_step=batch)

# # Predictions for the current training minibatch.
# train_prediction = tf.nn.softmax(logits)

# # Predictions for the test and validation, which we'll compute less often.
# eval_prediction = tf.nn.softmax(model(eval_data))