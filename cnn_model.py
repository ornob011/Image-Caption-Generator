from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.models import Model


def get_cnn_model():
    model = DenseNet201(weights='imagenet')
    fe = Model(inputs=model.input, outputs=model.layers[-2].output)

    return fe
