import optparse
import os

from display_images import display_images
from read_captions import read_captions
from caption_pre_processing import text_preprocessing
from img_features import get_img_features
from text_to_sequence import text_to_sequence
from encoder_decoder import image_caption_model
from datagen import get_datagenerator
from model_fit_parameter import save_model
from plot_train_vs_validation_loss import show_results


if __name__ == '__main__':

    parser = optparse.OptionParser()

    parser.add_option('-i', '--image_path', action="store", dest="image_path",
                      help="Input the image folder path", default="Flicker8k_Dataset/")

    parser.add_option('-c', '--caption_path', action="store", dest="caption_path",
                      help="Input the image caption path", default="captions.txt")

    parser.add_option('-g', '--gpu', action="store", dest="gpu",
                      help="If you want to use GPU, type yes otherwise no", default="yes")

    options, args = parser.parse_args()

    image_path = str(options.image_path)
    caption_path = str(options.caption_path)
    gpu = str(options.gpu)

    if (gpu == 'yes'):
        os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    data = read_captions(caption_path)
    display_images(image_path, data.sample(15))

    data = text_preprocessing(data)
    captions = data['caption'].tolist()

    tokenizer, max_length, vocab_size, train, test = text_to_sequence(
        captions, data)

    caption_model = image_caption_model(max_length, vocab_size)
    
    features = get_img_features(data, image_path)

    train_generator, validation_generator = get_datagenerator(train, 'image', 'caption', 64, image_path,tokenizer, vocab_size, max_length, features, test)

    checkpoint, earlystopping, learning_rate_reduction = save_model()

    history = caption_model.fit(train_generator, epochs=30, validation_data=validation_generator, callbacks=[checkpoint, earlystopping, learning_rate_reduction])

    show_results(history)
