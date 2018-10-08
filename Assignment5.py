#Camille Chow
#ECE 471 Assignment 5

import keras
from sklearn.model_selection import train_test_split

x_train = train["titles"].map(str) + " " + train["descriptions"].map(str)
y_train = train["labels"] - 1
x_test = test["titles"].map(str) + " " + test["descriptions"].map(str)
y_test = test["labels"] - 1

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)

def preprocess(data):
  t = keras.preprocessing.text.Tokenizer()
  t.fit_on_texts(data)
  encoded_text = t.texts_to_sequences(data)
  encoded_text = keras.preprocessing.sequence.pad_sequences(encoded_text, padding="post")
  return encoded_text

x_train = preprocess(x_train)
x_val = preprocess(x_val)
x_test = preprocess(x_test)

model = keras.Sequential()
