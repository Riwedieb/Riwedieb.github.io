---
layout: post
title: "How to build your own dog breed classifier"
date: "2020-05-30 15:01:08 +0200"
---

# Introduction
The project described in the following is part of Udacity’s Data Scientist nanodegree.
The main goal of the project is to develop an algorithm which classifies the breed of given dog images.
In case a picture of a human is given, the most lookalike dog breed shall be output.  

From Udacity, [this](https://github.com/udacity/dog-project/blob/master/dog_app.ipynb) Jupyter notebook was given as a template which guided through the development of the., in which a neural network shall implemented using keras and trained on a given dataset of dog pictures which are labeled with their breeds.   

The particular steps to be taken in the notebook are:  
1) [Import of the datasets](#import)  
2) [Detection of human faces](#Detect_Human)  
3) [Detection of dogs](#Detect_Dog)  
4) [A convolutional neural network (CNN) for classification from scratch](#Setup_CNN)  
5) [Training of an existing CNN for classification using transfer learning](#Existing_Transfer)  
6) [Setup & training of a new CNN for classification using transfer learning](#New_Transfer)  
7) Implementation of the whole classification algorithm  
8) Test of the algorithm  

## Import of the datasets <a name="import"></a>  
There are two datasets available, the first with 8351 dog images in total which are labeled with 133 dog breeds.
The dog dataset is split into 6680 images for training, 835 images for validating and 836 images for testing the algorithm.
The other dataset contains 13233 images of human faces.
Here is an example from each dataset:  
<p align="center">
  <img align="center" src="/images/Brittany_02625.jpg" width="225"/> <img src="/images/face_example.png" width="225"/>
</p>

## Detection of human faces <a name="Detect_Human"></a>
At first the algorithm shall check if it got an image of a human.
Hence in this step we try an already existing face recognition algorithm from the openCV library.
The algorithm decomposes gray-scale the image using Haar Wavelets at first and then leads found feature to a cascade of classifiers.
This is also called the [Viola-Jones method](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).
Testing the algorithm on a dataset of 100 dog images and 100 human face images, shows that:
* 100% of the human face images where correctly detected as humans
* 11% of the dog images where falsely detected as humans  

One backdraw of the Viola-Jones method is that it always needs a completely visible human face for a correct detection.
To avoid this problem, an app using the algorithm could simply tell the user if the face on the image is not clearly visible.
Another idea could be, to train the underlying model with images of people who's faces are not completely visible
so the model is better able to detect faces just by finding features like eyes, mouths or ears.

## Detection of dogs <a name="Detect_Dog"></a>  
After we have now an algorithm which can detect humans on an image, we need one more for checking for dogs.
In this case, a pre-trained [ResNet-50](http://ethereon.github.io/netscope/#/gist/db945b393d40bfa26006) model is used.
This model was trained on the [ImageNet](http://www.image-net.org/) dataset, which consists of over 10 million images with 1000 categories.
It takes square images of size 224x224 pixels with three color channels (R,G,B) as input,
and outputs an one hot encoded vector of 1000 elements length.
This [list](https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a) shows the label of each element.
Looking through the list we find, that the element indices 151-268 correspond to categories like 'Chihuahua' and 'Mexican hairless'.
This if ResNet-50 outputs a label between these indices, we assume the analyzed image shows a dog.  

How good does this work?
To find out the performance of the model,
the model was tested on the same data set made up of 100 dog and 100 human face images used already before.
The results are:
* 0% of the human face images where falsely detected as dogs
* 100% of the dog images where correctly detected as dogs  

## A convolutional neural network (CNN) for classification from scratch <a name="Setup_CNN"></a>
Since we now have the ability to find out if there is a dog or a human face on the picture,
we want to proceed further and guess which breed the dog or the human face is looking alike.
The CNN trained for this purpose is implemented using [Keras](https://keras.io/).
It has the following architecture:
...  TODO: Add Keras summary  

which was choosen after looking through the given example and several existing CNN architectures like e.g. [VGG-16](https://neurohive.io/en/popular-networks/vgg16/), [AlexNet](https://neurohive.io/en/popular-networks/alexnet-imagenet-classification-with-deep-convolutional-neural-networks/) and [ResNet](https://neurohive.io/en/popular-networks/resnet/) and trying several adaptions.
Like in the examples, the first convolutional layers + MaxPooling layers act as a feature recognition,
which should at first find small features like edges up to larger feature like the dog's eyes or jaws.
To not loose information, the number of kernels is doubled after each MaxPooling layer.
The features found by the CNN layers are then fed to dense layers which establishes the connection between features and dog breed.  
Except for the last layer, all activation functions are set to 'relu' since it is efficient to compute
and suppresses the "vanishing" gradients problem.
Since we have a classification problem, the activation function of the last layer is softmax.
The shown architecture tends to overfit fast, so the dataset is augmented using Keras' [ImageDataGenerator](https://keras.io/api/preprocessing/image/)
and a droput layer is inserted before the dense layer.

**Model summary:**  

|**Layer (type)**   |      **Output Shape**  | **Param #** |
|:-----------------:|:----------------------:|:--------:|
| Conv2D            |  (None, 224, 224, 4)   |   112    |
| Conv2D            |  (None, 224, 224, 4)   |   148    |
| MaxPooling2D      |  (None, 112, 112, 4)   |    0     |
| Conv2D            |  (None, 112, 112, 8)   |   296    |
| Conv2D            |  (None, 112, 112, 8)   |   584    |
| MaxPooling2D      |  (None, 56, 56, 8)     |    0     |
| Conv2D            |  (None, 56, 56, 16)    |   1168   |
| MaxPooling2D      |  (None, 28, 28, 16)    |    0     |
| Flatten           |  (None, 12544)         |    0     |
| Dropout           |  (None, 12544)         |    0     |
| Dense             |  (None, 200)           |  2509000 |  
| Dropout           |  (None, 200)           |    0     |
| Dense             |  (None, 200)           |  40200   |  
| Dropout           |  (None, 200)           |    0     |
| Dense             |  (None, 133)           |  26733   |  
|<img width=200px/> | <img width=200px/>     |<img width=200px/> |  

Total params: 2,578,241  
Trainable params: 2,578,241  
Non-trainable params: 0  

After training the model for 25 epochs, the achieved accuracy on the test dataset was 12.67%.


## Training of an existing CNN for classification using transfer learning <a name="Existing_Transfer"></a>
The training time was about 25 minutes on the GPU provided by Udacity while the achieved accuracy still
leaves much space for improvements.
Maybe more training time and a larger data set could improve the model performance.
Here the method of 'transfer learning' can help:  
At first, a pre-trained CNN, in this case the [VGG-16](https://neurohive.io/en/popular-networks/vgg16/) model is taken and
the last dense layers, mainly responsible for classification of the features detected by the CNN layers before, are cut away.
Then the remaining CNN is applied on the new dog dataset is and the outputs of the last layer are saved as so called 'bottleneck-features'.  
These bottleneck-features are then used to train a new dense layer, which replaces the former dense layer of the VGG-16 dataset.
The combination of pre-trained CNN and new dense layer can then form a new model with better performance than the one trained in in the section before.  
Here is a summary of the new dense layer:

**Model summary:**  
|**Layer (type)**   |      **Output Shape**  | **Param #** |
|:-----------------:|:----------------------:|:--------:|
| Global Average Pooling   |  (None, 512)      |       0     |
| Dense                    |  (None, 133)      |   68229     |
|<img width=200px/> | <img width=200px/>     |<img width=200px/> |  

Training the model just to about 40 seconds and while the accuracy on the test dataset is now 43.6% compared to 12.7% in our first try.
