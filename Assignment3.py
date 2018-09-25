#Camille Chow
#ECE 471 Assignment 3
import numpy as np
import tensorflow as tf

#dimensional constants
num_classes = 10
image_h = 28
image_w = 28
channels = 1
input_shape = (image_h, image_w, channels)
val_set_size = 10000

#tunable hyperparams
batch_size = 50
epochs = 2
kernel_size = 3
pool_size = 2
a_fcn = 'relu'
dropout = .35
lam = .001

#get data
mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#shape test data
x_test = x_test.reshape(x_test.shape[0], image_h, image_w, channels)
x_test = x_test.astype('float32')
x_test /= 255.0
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#shape training data
x_train = x_train.reshape(x_train.shape[0], image_h, image_w, channels)
x_train = x_train.astype('float32')
x_train /= 255.0
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

#split into training/validation sets
x_val = x_train[:val_set_size]
y_val = y_train[:val_set_size]
x_train = x_train[val_set_size:]
y_train = y_train[val_set_size:]

#build cnn
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=kernel_size, strides=(1, 1), activation=a_fcn, input_shape=input_shape))
model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides=pool_size))
model.add(tf.keras.layers.Conv2D(64, kernel_size=kernel_size, activation=a_fcn))
model.add(tf.keras.layers.MaxPooling2D(pool_size=pool_size))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(500, activation=a_fcn, kernel_regularizer=tf.keras.regularizers.l2(lam)))
model.add(tf.keras.layers.Dropout(dropout))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

#train model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))
#model.fit(x_val, y_val, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

#test model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])