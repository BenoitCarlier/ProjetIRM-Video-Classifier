from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
import numpy as np

# Definition du model à partir de InceptionV3 qui va nous permettre d'extraire les features de chaque frame

inception = InceptionV3(weights='imagenet',include_top=True)

feature_extractor = Model(
            inputs=inception.inputs,
            outputs=inception.get_layer('avg_pool').output
            )


def extract_features(image):
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    features = feature_extractor.predict(image)
    features = features[0]
    return(features)
# Lecture des videos une par une et 'pickle dumping' de la séquence des features dans un fichier par vidéo

import pickle
import imageio
import os
import time
import utils
import numpy as np
from scipy import misc

path_to_videos = 'UCF-101'
path_to_features = 'data/Features'

frames_number = 40

if not os.path.isdir(path_to_features):
    os.mkdir(path_to_features)

categories = os.listdir(path_to_videos)

def rescale_list(array, size): # Resize a 40 frames pour toutes les vidéos
    length = len(array)
    if length < size:
        return None
    skip = length // size
    output = np.array([array[i] for i in range(0, length, skip)])
    return output[0:size]

def store_features(category):
    print('processing category : ', category)
    dir_name = os.path.join(path_to_videos, category)
    video_list = os.listdir(dir_name)
    print(video_list)
    store_dir_name = os.path.join(path_to_features, category)
    if not os.path.isdir(store_dir_name):
        os.mkdir(store_dir_name)
    for video in video_list:
        print('processing video : ', video)
        path_to_video = os.path.join(dir_name, video)
        feature_file_name = video.replace('.avi', '.pkl')
        path_to_feature = os.path.join(store_dir_name, feature_file_name)
        if os.path.isfile(path_to_feature):
            print('file ', video, ' features are already stored')
        else:
            try:
                video_file = imageio.get_reader(path_to_video)
                frames = [frame for frame in video_file]
                frames = rescale_list(frames, frames_number)
                sequence = []
                for frame in frames:
                    image = misc.imresize(frame,(299,299))
                    features = extract_features(image.astype(np.float64))
                    sequence.append(features)
                with open(path_to_feature, 'wb') as file:
                    pickle.dump(sequence, file)
                    print(file, 'features stored')
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
    return()

start = time.time()
for category in categories:
    store_features(category)
    checkpoint = time.time()
    print('---------------------*****---------------------')
    print(category + ' done!')
    print('Time elapsed: {}s'.format(checkpoint - start))
    print('---------------------*****---------------------')
