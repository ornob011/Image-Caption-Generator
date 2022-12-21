from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from cnn_model import get_cnn_model
import os
import pickle


def get_img_features(data, image_path):

    fe = get_cnn_model()

    img_size = 224
    features = {}

    for image in tqdm(data['image'].unique().tolist()):
        img = load_img(os.path.join(image_path, image),
                       target_size=(img_size, img_size))
        img = img_to_array(img)
        img = img/255.

        img = np.expand_dims(img, axis=0)

        feature = fe.predict(img, verbose=0)

        features[image] = feature

    with open('image_feature_extracted.pkl', 'wb') as handle:
        pickle.dump(features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('image_feature_extracted.pkl', 'rb') as handle:
        features = pickle.load(handle)
    
    return features
