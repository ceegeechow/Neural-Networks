#Camille Chow
#ECE 471 Assignment 4
#Classifying CIFAR10 data
#Citation: Learning Multiple Layers of Features from Tiny Images, Alex Krizhevsky, 2009.
import numpy as np
import tensorflow as tf

#dimensional constants
num_classes = 10
image_h = 32
image_w = 32
channels = 3

#get data
(x_train, y_train),(x_test, y_test) = tf.keras.datasets.cifar10.load_data()

#shape test data
x_test = x_test.reshape(x_test.shape[0], image_h, image_w, channels)
x_test = x_test.astype('float32')
x_test /= 255.0
# shape_data(x_test)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

#shape training data
x_train = x_train.reshape(x_train.shape[0], image_h, image_w, channels)
x_train = x_train.astype('float32')
x_train /= 255.0

# shape_data(x_train)
y_train = tf.keras.utils.to_categorical(y_train, num_classes)

#tunable hyperparams
batch_size = 100
epochs = 100
lam = .001

def lr_schedule(epoch):
    lrate = 0.001
    if epoch > 65:
        lrate = 0.0005
    elif epoch > 85:
        lrate = 0.0001
    return lrate

def add_conv_layer(model, num_filters):
	model.add(tf.keras.layers.Conv2D(num_filters, kernel_size=3, 
		strides=(1, 1), activation='elu', padding='same'))

def add_pooling_layer(model):
	model.add(tf.keras.layers.MaxPooling2D(pool_size=3, strides=2))

def add_bn_layer(model):
	model.add(tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=0.001))

#build cnn
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=3, 
		strides=(1, 1), activation='elu', padding='same', input_shape=(image_h, image_w, channels)))
add_bn_layer(model)
add_conv_layer(model, 32)
add_bn_layer(model)
add_pooling_layer(model)
model.add(tf.keras.layers.Dropout(.2))
add_conv_layer(model, 64)
add_bn_layer(model)
add_conv_layer(model, 64)
add_bn_layer(model)
add_pooling_layer(model)
model.add(tf.keras.layers.Dropout(.3))
add_conv_layer(model, 128)
add_bn_layer(model)
add_conv_layer(model, 128)
add_bn_layer(model)
add_pooling_layer(model)
model.add(tf.keras.layers.Dropout(.4))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256, activation='elu',kernel_regularizer=tf.keras.regularizers.l2(lam)))
model.add(tf.keras.layers.Dropout(.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))

#train model
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=.1, callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_schedule)])

#test model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])