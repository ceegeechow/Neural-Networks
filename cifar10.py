#Camille Chow
#ECE 471 Assignment 4
#Classifying CIFAR10 data
#Citation: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# from keras.models import Sequential
# from keras.layers.convolutional import Convolution2D, MaxPooling2D
# from keras.layers import Activation, Flatten, Dense, Dropout
# from keras.layers.normalization import BatchNormalization

#dimensional constants
num_classes = 10
image_h = 32
image_w = 32
channels = 3
val_set_size = 5000

#get data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()

def shape_data(data):
  data = data.reshape(data.shape[0], image_h, image_w, channels)
  data = data.astype('float32')
  data /= 255.0
  data -= 0.5

#shape test data
x_test = x_test.reshape(x_test.shape[0], image_h, image_w, channels)
x_test = x_test.astype('float32')
x_test /= 255.0
x_test -= 0.5
# shape_data(x_test)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#shape training data
x_train = x_train.reshape(x_train.shape[0], image_h, image_w, channels)
x_train = x_train.astype('float32')
x_train /= 255.0
x_train -= 0.5
# shape_data(x_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

# x_val = x_train[:val_set_size]
# y_val = y_train[:val_set_size]
# x_train = x_train[val_set_size:]
# y_train = y_train[val_set_size:]

#shuffle data
# indices = np.random.permutation(50000)
# x_train = x_train[indices,:]
# y_train = y_train[indices,:]

# image_gen = ImageDataGenerator(
# #     featurewise_center=True,
# #     featurewise_std_normalization=True,
#     rescale=1./255,
#     rotation_range=15,
#     width_shift_range=.15,
#     height_shift_range=.15,
#     horizontal_flip=True)

# #training the image preprocessing
# image_gen.fit(x_train, augment=True, rounds=10)

# # WeightFunction = lambda x : 1./x**0.75
# # ClassLabel2Index = lambda x : lohe.le.inverse_tranform( [[x]])
# # CountDict = dict( df["Id"].value_counts())
# # class_weight_dic = { lohe.le.transform( [image_name])[0] : WeightFunction(count) for image_name, count in CountDict.items()}
# # del CountDict


#tunable hyperparams
batch_size = 100
epochs = 10
dropout = .35
dense_units = 1000
lam = .003

def add_conv_layer(model, num_filters):
	model.add(tf.keras.layers.Conv2D(num_filters, kernel_size=3, 
		strides=(1, 1), activation='elu', padding='same'))

def add_pooling_layer(model):
	model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2))

def add_bn_layer(model):
	model.add(tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001))

#build cnn
model = tf.keras.Sequential()
# add_conv_layer(model, 48, input_shape=(image_h, image_w, channels))
model.add(tf.keras.layers.Conv2D(48, kernel_size=3, 
		strides=(1, 1), activation='elu', padding='same', input_shape=(image_h, image_w, channels)))
add_bn_layer(model)
add_conv_layer(model, 48)
add_bn_layer(model)
add_pooling_layer(model)
model.add(tf.keras.layers.Dropout(.2))
add_conv_layer(model, 96)
add_bn_layer(model)
add_conv_layer(model, 96)
add_bn_layer(model)
add_pooling_layer(model)
model.add(tf.keras.layers.Dropout(.3))
add_conv_layer(model, 192)
add_bn_layer(model)
add_conv_layer(model, 192)
add_bn_layer(model)
add_pooling_layer(model)
model.add(tf.keras.layers.Dropout(.4))
model.add(tf.keras.layers.Flatten())
# model.add(tf.keras.layers.Dense(512, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(lam)))
# model.add(tf.keras.layers.Dropout(.5))
# model.add(tf.keras.layers.Dense(256, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(lam)))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

#train model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=.1)
# model.fit_generator(image_gen.flow(x_train, y_train, batch_size=batch_size),
#           steps_per_epoch=  x_train.shape[0]//batch_size,
#           epochs=epochs,
#           verbose=1,
#           validation_data=(x_val,y_val))


#test model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])