import pickle
import os
from keras.layers import LSTM, Dense
from keras.models import Sequential
import numpy as np

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
    class_seq = []
    categories = os.listdir(path_to_features)
    class_nb = 0
    for category in categories:
        print('category : ', category)
        sequences = os.listdir(os.path.join(path_to_features, category))
        for seq_file in sequences:
            path_to_seq_file = os.path.join(path_to_features, category, seq_file)
            seq = pickle.load(open(path_to_seq_file, 'rb'))
            total_seq.append(seq)
            class_seq.append(class_nb)
        class_nb += 1
        print(class_nb)
    total_seq = np.array(total_seq)
    class_seq = np.array(class_seq)
    return(total_seq, class_seq)

print(np.shape(load_features()[0]))
print(np.shape(load_features()[1]))