import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

NUM_SAMP = 1000

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
    return self.data, self.labels

# class Data():
#   def __init__(self):
#     np.random.seed()
#     self.t = np.atleast_2d(np.arange(NUM_SAMP)/NUM_SAMP).T
    
#     self.w1 = 50*np.random.randn()
#     self.phi1 = np.random.randn()
#     self.sine1 = np.atleast_2d(np.sin(self.w1*self.t + self.phi1))
#     self.b1 = 10*np.random.randn()
#     self.line1 = 2*self.t + self.b1
#     self.coords1 = np.hstack((self.t, self.sine1))
    
#     self.w2 = 50*np.random.randn()
#     self.phi2 = np.random.randn()
#     self.sine2 = np.atleast_2d(np.sin(self.w2*self.t + self.phi2))
#     self.b2 = 10*np.random.randn()
#     self.line2 = 2*self.t + self.b2
#     self.coords2 = np.hstack((self.t, self.sine2))
    
#     self.coords = np.vstack((self.coords1, self.coords2))
#     self.labels = np.atleast_2d(np.hstack((np.zeros(NUM_SAMP), np.ones(NUM_SAMP)))).T
    
#   def get_data(self):
#     return self.coords, self.labels

# class Data():
#   def __init__(self):
#     np.random.seed()
#     self.x = np.random.normal(size=(128, NUM_SAMP))
#     self.w = np.random.normal(size=(128, 1))
#     self.labels = np.sign(np.matmul((self.x).T, self.w))
#   def get_data(self):
#     return self.x.T, self.labels

data = Data()
all_points, all_labels = data.get_data() 

plt.plot(data.x, data.g1)
plt.plot(data.x, data.g2)

# plt.plot(all_points[:1000,0], all_points[:1000,1])

data = Data()
all_points, all_labels = data.get_data()
threshold = .35
Possible_Failures = 12

num_layers = 32
layers = []
probes = []
losses = []
optims = []

tf.reset_default_graph()

points = tf.placeholder(tf.float32, [None, 2])
choice = tf.placeholder(tf.float32, [None, 1])
varlist = []


def my_leaky_relu(x):
  return tf.nn.leaky_relu(x, alpha=.5)

for i in range (num_layers):
  if i == 0:
    layers.append(tf.layers.dense(points, 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), trainable=False))
    
    
  else:
    layers.append(tf.layers.dense(layers[i-1], 128, activation=my_leaky_relu, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), trainable=False))
    
  probes.append(tf.layers.dense(layers[i], 1, activation=None, kernel_initializer = tf.glorot_normal_initializer(seed = None, dtype=tf.float32), name=("Probe"+str(i))))
  
  varlist.append(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ("Probe" + str(i))))
  
  losses.append(tf.losses.sigmoid_cross_entropy(choice, probes[i]))
  optims.append(tf.train.RMSPropOptimizer(learning_rate = 0.1, momentum=0.75).minimize(losses[i], var_list=varlist[i]))
 
                

# layer = tf.layers.dense(points, 128, activation=my_leaky_relu)
# probe = tf.layers.dense(layer, 1, activation=None, name=("Probe"))
# varlist = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, ("Probe"))
# loss = tf.losses.sigmoid_cross_entropy(choice, probe)
# optim = tf.train.RMSPropOptimizer(learning_rate=.25, momentum = 0.7).minimize(loss, var_list=varlist)

init = tf.global_variables_initializer()


# sess = tf.Session()

# sess.run(init)
# for _  in range(0, 300):
#   loss_np, _ = sess.run([loss, optim], feed_dict={points: all_points, choice: all_labels})
  

# print(loss_np)
# probe_results, choice_vals = (sess.run([tf.round(tf.nn.sigmoid(probe)), choice], feed_dict={points: all_points, choice: all_labels}))
# #print(probe_results)
# probe_error = np.sum(np.round(np.abs(probe_results-choice_vals)))/(2*NUM_SAMP)
# #print(probe_error)

probe_error_experiment = []

experiment_number = 0

while experiment_number < 10:
  sess = tf.Session()
  sess.run(init)
  Failure_Counter = 0
  
  print("experiment #: ", experiment_number)
  data = Data()
  all_points, all_labels = data.get_data()
  probe_error = []
  
  layers_number = 0
  while layers_number < num_layers-1:
    for _  in range(0, 100):
      loss_np, _ = sess.run([losses[layers_number], optims[layers_number]], feed_dict={points: all_points, choice: all_labels})
#       if (layers_number == 0):
#         print(loss_np)
    if layers_number == 0 and loss_np > threshold:
      Failure_Counter = Failure_Counter + 1
      if Failure_Counter == Possible_Failures:
          print("Dataset Failed")
          layers_number = 0
          experiment_number -= 1
          break
    else:
      probe_results, choice_vals = (sess.run([tf.round(tf.nn.sigmoid(probes[layers_number])), choice], feed_dict={points: all_points, choice: all_labels}))
      probe_error.append (np.sum(np.round(np.abs(probe_results-choice_vals)))/(2*NUM_SAMP))
      layers_number += 1
      Failure_Counter = 0
      print("Layer", layers_number, "Loss", loss_np)
    
#     if loss_np < threshold+0.01*layers_number:
#         probe_results, choice_vals = (sess.run([tf.round(tf.nn.sigmoid(probes[layers_number])), choice], feed_dict={points: all_points, choice: all_labels}))
#         probe_error.append (np.sum(np.round(np.abs(probe_results-choice_vals)))/(2*NUM_SAMP))
#         layers_number += 1
#         Failure_Counter = 0
#         print("Layer", layers_number, "Loss", loss_np)
#     else:
#       Failure_Counter = Failure_Counter + 1
#       if Failure_Counter == Possible_Failures:
#           print("Dataset Failed")
#           layers_number = 0
#           experiment_number -= 1
#           break
    
  experiment_number += 1
  if Failure_Counter != Possible_Failures:
    print(probe_error)
    probe_error_experiment.append(probe_error) 
  sess.close()
    
probe_error_experiment = np.asarray(probe_error_experiment)
average_layer = np.mean(probe_error_experiment, axis=0)

print(average_layer)


# print(probe_error)
x = np.arange(1,num_layers)
plt.bar(x, average_layer, tick_label=x)



#print(blah)

#print(all_labels)