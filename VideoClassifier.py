import pickle
import keras
import os
from keras.layers import LSTM, Dense
from keras.models import Sequential

path_to_features = 'data/Features'

lstm_model = Sequential()

lstm_model.add(LSTM(256,input_shape=(40,2048)))
lstm_model.add(Dense(101, activation='softmax'))

lstm_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

def load_features():
    total_seq = []
    categories = os.listdir(path_to_features)