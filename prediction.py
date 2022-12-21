import optparse
import os
import pickle
import numpy as np
from tqdm import tqdm

from display_images import display_images
from read_captions import read_captions
from caption_pre_processing import text_preprocessing
from text_to_sequence import text_to_sequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


def idx_to_word(integer, tokenizer):

    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def predict_caption(model, image, tokenizer, max_length, features):

    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)

        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)

        word = idx_to_word(y_pred, tokenizer)

        if word is None:
            break

        in_text += " " + word

        if word == 'endseq':
            break

    return in_text


if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-i', '--image_path', action="store", dest="image_path",
                      help="Input the image folder path", default="Flicker8k_Dataset/")

    parser.add_option('-c', '--caption_path', action="store", dest="caption_path",
                      help="Input the image caption path", default="captions.txt")

    parser.add_option('-g', '--gpu', action="store", dest="gpu",
                      help="If you want to use GPU, type yes otherwise no", default="yes")

    parser.add_option('-f', '--feature_filepath', action="store", dest="feature_filepath",
                      help="Input the pickled image features path", default="image_feature_extracted.pkl")

    options, args = parser.parse_args()

    image_path = str(options.image_path)
    caption_path = str(options.caption_path)
    gpu = str(options.gpu)
    feature_filepath = str(options.feature_filepath)

    if (gpu == 'yes'):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    data = read_captions(caption_path)

    data = text_preprocessing(data)
    captions = data['caption'].tolist()

    tokenizer, max_length, vocab_size, train, test = text_to_sequence(
        captions, data)

    with open(feature_filepath, 'rb') as handle:
        features = pickle.load(handle)

    samples = test.sample(15)
    samples.reset_index(drop=True, inplace=True)

    caption_model = load_model('model.h5')

    for index, record in tqdm(samples.iterrows()):

        caption = predict_caption(
            caption_model, record['image'], tokenizer, max_length, features)
        caption = caption.replace('startseq', '')
        caption = caption.replace('endseq', '')

        samples.loc[index, 'caption'] = caption

    display_images(image_path, samples, test=True)
