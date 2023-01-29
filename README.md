# Project Title
Image Caption Generator Model using Encoder-Decoder Architecture

Model code of the [Image Caption Generator Interface](https://github.com/ornob011/Image-Caption-Interface)

## Table of Contents
- [Project Title](#project-title)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Contributing](#contributing)
  - [License](#license)

<div style="text-align: justify;">

## Introduction

An image caption generator is a computer vision model that takes in an image as input and generates a natural language description of the scene depicted in the image. The goal of an image caption generator is to produce a human-like textual description of an image, which can be useful in a variety of applications, such as creating alternative text for images in websites or generating captions for images in photo albums.

There are several approaches to building an image caption generator, but the implemented approach in this solution involved using a pre-trained CNN (DenseNet201) to extract features from the image and LSTM to generate the corresponding caption. The CNN is responsible for extracting features such as edges, shapes, and textures from the image, while the LSTM processes these features and generates a sequence of words to describe the image. The LSTM can be trained on a large dataset of images and their corresponding captions to learn how to generate descriptive and accurate captions.

There are many challenges in building an image caption generator, including the need to accurately describe the contents of the image, handle variations in language, and generate coherent and grammatically correct sentences. The quality of the generated captions depended on the quality and diversity of the training data.

## Features

- Shows how Image Captioning problems can be approached
- Easy to understand the code

## Installation

To run the model, follow these steps:

1. Clone the repository to your local machine:
   
   ```
   git clone https://github.com/ornob011/Image-Caption-Generator.git
   ```

2. Navigate to the project directory:

    ```
    cd Image-Caption-Generator
    ```

3. Install the dependencies (for non-GPU):

    ```
    pip install -r requirements.txt
    ```

    Install the dependencies (for GPU): 

    ```
    pip install -r requirements-gpu.txt
    ```

4. Download the dataset from this link:
   ```
   https://www.kaggle.com/datasets/adityajn105/flickr8k
   ```
## Usage

This `Usage` section provides detailed instructions on how to use the application, including how to run it, import it into your code, and use its various functions.

To train the model, run the following command in a terminal:
   
   ```
   python train_model.py --image_path Your_Image_Folder_Path --caption_path Your_Caption_Path
   ```


To predict caption from images with the trained model, run the following command in a terminal:
   
   ```
   python prediction.py --image_path Your_Image_Folder_Path --caption_path Your_Caption_Path
   ```
## Contributing

I welcome contributions to this project! If you have an idea for a feature or improvement, or if you have found a bug, please feel free to open an issue in the [issue tracker](https://github.com/ornob011/Image-Caption-Generator/issues).

Before submitting a pull request, please make sure to:

- Read and follow our [contribution guidelines](CONTRIBUTING.md).
- Test your changes thoroughly.

Thank you for your contribution!


## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)  

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

MIT License

Copyright (c) 2022 Ornob Rahman

</div>