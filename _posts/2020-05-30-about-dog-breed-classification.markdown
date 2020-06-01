---
layout: post
title: "How to build your own dog breed classifier"
date: "2020-05-30 15:01:08 +0200"
---
The project described in the following is part of Udacity’s Data Scientist nano-degree and hosted in [**this**](https://github.com/Riwedieb/dog_app) github repository.
The main goal of the project is to develop an algorithm which classifies the breed of given dog images.
In case a picture of a human is given, the most lookalike dog breed shall be output.  

<style>
table{
    border-collapse: collapse;
    border-spacing: 0;
    border:2px solid #ffffff;
}

th{
    border:2px solid #000000;
}

td{
    border:1px solid #000000;
}

table th:first-of-type {
    width: 10%;
}
table th:nth-of-type(2) {
    width: 10%;
}
table th:nth-of-type(3) {
    width: 10%;
}
table th:nth-of-type(4) {
    width: 10%;
}
</style>

## Introduction

From Udacity, [this](https://github.com/udacity/dog-project/blob/master/dog_app.ipynb) Jupyter notebook was given as a template which guided through the development of the dog breed classifier, in which a neural network shall be implemented using keras and trained on a given dataset of dog pictures which are labeled with their breeds.   

The particular steps to be taken in the notebook are:  

1) [Import of the datasets](#import)  
2) [Detection of human faces](#Detect_Human)  
3) [Detection of dogs](#Detect_Dog)  
4) [A convolutional neural network (CNN) for classification from scratch](#Setup_CNN)  
5) [Training of an existing CNN for classification using transfer learning](#Existing_Transfer)  
6) [Training of a new CNN for classification using transfer learning](#New_Transfer)  
7) [Implementation of the classification algorithm](#Implementation)  
8) [Test of the algorithm](#Test)  

## Import of the datasets <a name="import"></a>  
There are two datasets available, the first with 8351 dog images in total which are labeled with 133 dog breeds.
The dog dataset is split into 6680 images for training, 835 images for validating and 836 images for testing the algorithm.
The other dataset contains 13233 images of human faces.
Here is an example from each dataset:  
<p align="center">
  <img align="center" src="/images/Brittany_02625.jpg" width="225"/> <img src="/images/face_example.png" width="225"/>

  <figcaption align="center">
  Fig.1 Left: example of a dog in the dataset. Right: Example of a human face in the dataset.
  </figcaption>
</p>

## Detection of human faces <a name="Detect_Human"></a>
At first the algorithm shall check if it got an image of a human.
Hence in this step we try an already existing face recognition algorithm from the openCV library.
The algorithm decomposes gray-scale images using Haar Wavelets at first and then leads found feature to a cascade of classifiers.
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



**Model summary:**  

| &nbsp;&nbsp;&nbsp;&nbsp; **Layer (type)** &nbsp;&nbsp;&nbsp;&nbsp;  |      **Output Shape**  | **Param #** |
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

Total params: 2,578,241  
Trainable params: 2,578,241  
Non-trainable params: 0  

The architecture was choosen after looking through the given example and several existing CNN architectures like e.g. [VGG-16](https://neurohive.io/en/popular-networks/vgg16/), [AlexNet](https://neurohive.io/en/popular-networks/alexnet-imagenet-classification-with-deep-convolutional-neural-networks/) and [ResNet](https://neurohive.io/en/popular-networks/resnet/) and trying several adaptions.
Like in the examples, the first convolutional layers + MaxPooling layers act as a feature recognition,
which should at first find small features like edges up to larger feature like the dog's eyes or jaws.
To not loose information, the number of kernels is doubled after each MaxPooling layer.
The features found by the CNN layers are then fed to dense layers which establishes the connection between features and dog breed.  
Except for the last layer, all activation functions are set to 'relu' since it is efficient to compute
and suppresses the "vanishing" gradients problem.
Since we have a classification problem, the activation function of the last layer is softmax.
The shown architecture tends to overfit fast, so the dataset is augmented using Keras' [ImageDataGenerator](https://keras.io/api/preprocessing/image/)
and a droput layer is inserted before the dense layer.  
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

Total params: 68,229  
Trainable params: 68,229  
Non-trainable params: 0  

Training the model just to about 40 seconds and while the accuracy on the test dataset is now 43.6% compared to 12.7% in our first try.

## Training of a new CNN for classification using transfer learning  <a name="New_Transfer"></a>
In this section a new model will be designed and trained using transfer learning
to further improve the accuracy above 60% on the test dataset.
Here, not just bottleneck features of one pre-trained model are given but four.
You can download the bottleneck features from these links:
- [VGG-19](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogVGG19Data.npz) bottleneck features
- [ResNet-50](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogResnet50Data.npz) bottleneck features
- [Inception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogInceptionV3Data.npz) bottleneck features
- [Xception](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/DogXceptionData.npz) bottleneck features

According to the task definition, just bottleneck features of one model should be used for training,
but out of curiosity I decided to evaluate all four models.  
I checked which output layers architecture the pre-trained CNNs use, see e.g. [InceptionV3](https://software.intel.com/content/www/us/en/develop/articles/inception-v3-deep-convolutional-architecture-for-classifying-acute-myeloidlymphoblastic.html).
These CNNs use 2-3 dense layers at their outputs so it was obvious to decide for a similar architecture,
in this case three dense layers.
I tried different dense layer sizes like e.g. 1024-2048 nodes which where used in the
pre-trained models but it turned out that 512 nodes gave the best results.
Depending on whether the bottleneck features had a 2D shape (VGG19, Xception, InceptionV3) or not (ResNet-50)
I used GlobalAveragePooling2D() (GAP) or Flatten(), respectively, as interfacing layer:

**Model summary:**  

| **Layer (type)**  |      **Output Shape**  | **Param #** |
|:--------------------------------:|:-----------------:|:--------:|
| Global Average Pooling / Dense   |  (None, 512)      |       0     |   
| Dropout                          |  (None, 512)      |       0     |   
| Dense                            |  (None, 512)      |   262656    |
| Dropout                          |  (None, 512)      |       0     |
| Dense                            |  (None, 512)      |   262656    |
| Dropout                          |  (None, 512)      |       0     |
| Dense                            |  (None, 133)      |   68229     |

Total params: 593,541  
Trainable params: 593,541  
Non-trainable params: 0  

After the training has finished, all models where evaluated on test data.
Here are the resulting accurracies:
- VGG19: 73.9%
- ResNet-50: 79.3%
- InceptionV3: 81,1%
- Xception: 83.9%

So the Xception model is taken for further procedure.  
The last step of the section is to implement the prediction function `breed_predictor()`,
where the pre-trained Xception model is combined with the new output layers.
The function then simply takes a path to an image and returns the three most probable
dog breeds together with their estimated probabilities.

## Implementation of the whole classification algorithm <a name="Implementation"></a>
For the use in e.g. a web app,
the prediction function shall at first determine whether a given image contains a human, dog or neither.
In case a dog a or a human is found, the most likely breeds shall be returned.
If neither is detected, the function should return an error message.  
The function `dog_human_breed_predictor()` implements these requirements.
It takes the functions `face_detector()` and `dog_detector()` implemented in the sections
before for detecting dogs and human faces.
If they respond positive, the `breed_predictor()` function is called.
Its return values, the three most probable dog breeds together with their estimated probabilities,
are then printed to screen.

## Test of the algorithm <a name="Test"></a>
In this last section, the behavior of `dog_human_breed_predictor()` is evaluated for several test images.
At first we try to predict the breeds of three dog images, see Fig. 2.

<p align="center">
  <img align="center" src="/images/Schaeferhund.png" width="300"/>
  <img src="/images/Labrador_retriever.png" width="300"/>
  <img src="/images/Brittany.png" width="300"/>
  <figcaption align="center">
  Fig. 2 From left to right: German Shepherd, Labrador Retriever, Brittany
  </figcaption>
</p>

All three images where classified as dogs.
The three highest probabilities for the breeds predicted by the function for the images are:

|  **Image**         | **First**               | **Second**           | **Third**                  |
|:------------------:|:-----------------------:|:--------------------:|:--------------------------:|
| German Shepherd    | German shepherd, 99%    | Belgian malinois, 0% | Belgian tervuren, 0%       |
| Labrador Retriever | Labrador retriever, 100%| Pointer, 0%          | Anatolian shepherd dog, 0% |
| Brittany           | Brittany, 100%    |Irish red and white setter, 0% | Cavalier king charles spaniel. 0%

Looks like the dog breed detection works quite good, so let's try it on images of humans.
<p align="center">
  <img align="center" src="/images/Bean1.png" width="300"/>
  <img src="/images/Bean2.png" width="300"/>
  <img src="/images/Nielsen1.png" width="300"/>

 <figcaption align="center">
  Fig. 3 From left to right: Mr. Bean, Mr. Bean with bear, Lt. Frank Drebin.
 </figcaption>
</p>

All three images where classified as human faces.
The three highest probabilities for the breeds predicted by the function for the images are:

|  **Image**         | **First**               | **Second**           | **Third**                  |
|:------------------:|:-----------------------:|:--------------------:|:--------------------------:|
| Mr. Bean    | Entlebucher mountain dog, 34%  | Poodle, 10%          | Australian cattle dog, 10%       |
| Mr. Bean w. bear | Chinese crested, 56%      | Poodle, 43%          | Cavalier king charles spaniel, 0% |
| L. Nielsen       | Afghan hound, 80%         | Poodle, 17%          | Chinese crested 4% |

And the last set of images is intended to confuse the algorithm a bit :-)  

<p align="center">
  <img align="center" src="/images/cat.png" width="300"/>
  <img src="/images/fish.png" width="300"/>
  <img src="/images/woman.png" width="300"/>

 <figcaption align="center">
  Fig. 4 From left to right: Cat, Lt. Frank Drebin with fish, woman with dogs.
 </figcaption>
</p>

Again the results are:  

|  **Image**         | **First**               | **Second**           | **Third**                  |
|:------------------:|:-----------------------:|:--------------------:|:--------------------------:|
| Cat                | French bulldog, 84%     | Cane corso, 6%       | Curly-coated retriever, 3% |
| F. Drebin with fish | NA                      | NA                   | NA                         |
| Woman with dogs    | Afghan hound, 100%      | Leonberger, 0%       | Borzoi 0%                  |

In the case of Lt. Frank Drebin,
the function could not find a dog or human face on the image.

### Discussion

*Is the output better than expected? Or worse?*  
In case breeds are estimated on dog images, I'm impressed by the performance of the model. The models output vector shows that in the first two cases (Labrador Retriever and Brittany) the model is very confident in it's decision, all other elements are close to zero. The image with the German Shepherd is as I guess a bit harder to interpret, because of the branch in the dogs mouth and the breed can also by humans easily be confused with similar looking a Belgian tervuren and Belgian malinois. This is visible in the output vector, where the model estimates the breed to German Shepherd with 99% probability but also the Belgian Tervuren and the Belgian malinois appears as next probable category.  

For images of humans the model doesn't seem to create a stable output for each individual. There are two images of Mr. Bean where the model one time predicted a Entlebucher mountain dog and for the other image an Chinese crested dog. The reason for this is unclear.

Regarding the cat image i am wondering why the `face_detector()` triggered.
A reason could be because it uses haar wavelets which may create a characteristic signal when convoluted with the cat's tiger stripes(?)
Could be interesting for further research.

*What could be ideas for improvements:*  
1) If multiple persons and/or dogs are visible, they should be labeled seperately. E.g. in the last picture, two dogs and one woman are visible. The woman ist detected by the `face_detector()` function, while the dog breed is most probably predicted from the two dogs to her side. A first model could be used to extract the region in a picture where there is (mostly) just a human or dog visible and then passes it to the `breed_predictor()` function.  
2) For a better seperation of humans and dogs the outputs probabilities of face_detector and dog_detector could be compared to choose the one with the higher activation signal.  
3) The face_detector() which uses Haar Wavelets could be replaced by a CNN to make it less prone to false positives on e.g. cats, see example below.  
4) More time could be invested in fine tuning the networks architecture e.g. different number of dense layers and different dense layer sizes.
