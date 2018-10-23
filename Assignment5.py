#Camille Chow
#ECE 471 Assignment 5
import pandas as pd
import keras
from sklearn.model_selection import train_test_split

train = pd.read_csv('ag_news_csv/train.csv', names=["labels", "titles", "descriptions"])
test = pd.read_csv('ag_news_csv/test.csv', names=["labels", "titles", "descriptions"])

max_len = 400 #chosen based on max length = 1012, avg length = 237
num_classes = 4

x_train = train["titles"].map(str) + " " + train["descriptions"].map(str)
y_train = train["labels"] - 1
x_test = test["titles"].map(str) + " " + test["descriptions"].map(str)
y_test = test["labels"] - 1

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

t = keras.preprocessing.text.Tokenizer()
t.fit_on_texts(x_train)
x_train = t.texts_to_sequences(x_train)
x_val = t.texts_to_sequences(x_val)
x_test = t.texts_to_sequences(x_test)

x_train = keras.preprocessing.sequence.pad_sequences(x_train, padding="post", truncating="post", maxlen=max_len)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, padding="post", truncating="post", maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, padding="post", truncating="post", maxlen=max_len)

y_train = keras.utils.to_categorical(y_train, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

epochs = 1
batch_size = 100

model = keras.Sequential()
model.add(keras.layers.Embedding(len(t.word_counts), 40, input_length=max_len))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_val, y_val))

#test model
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])