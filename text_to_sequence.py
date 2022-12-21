from tensorflow.keras.preprocessing.text import Tokenizer
import pickle


def text_to_sequence(captions, data):

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(captions)
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(len(caption.split()) for caption in captions)

    images = data['image'].unique().tolist()
    nimages = len(images)

    split_index = round(0.8*nimages)
    train_images = images[:split_index]
    val_images = images[split_index:]

    train = data[data['image'].isin(train_images)]
    test = data[data['image'].isin(val_images)]

    train.reset_index(inplace=True, drop=True)
    test.reset_index(inplace=True, drop=True)

    tokenizer.texts_to_sequences([captions[1]])[0]

    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return tokenizer, max_length, vocab_size, train, test
