from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from textwrap import wrap


def readImage(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img)
    img = img/255.

    return img


def display_images(image_path, temp_df, test=False):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(15):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = readImage(f"{image_path}{temp_df.image[i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")

        if (test == False):
            plt.savefig('random_image_with_caption.png')
        elif (test == True):
            plt.savefig('inference_with_generated_caption.png')
