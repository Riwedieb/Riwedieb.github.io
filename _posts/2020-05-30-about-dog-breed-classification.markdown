---
layout: post
title: "How to build your own dog breed classifier"
date: "2020-05-30 15:01:08 +0200"
---

### Introduction

The project described in the following is part of Udacity’s Data Scientist nanodegree.
The main goal of the project is to develop an algorithm which classifies the breed of given dog images.
In case a picture of a human is given, the most lookalike dog breed shall be output.  

From Udacity, [this](https://github.com/udacity/dog-project/blob/master/dog_app.ipynb) Jupyter notebook was given as a template which guided through the development of the., in which a neural network shall implemented using keras and trained on a given dataset of dog pictures which are labeled with their breeds.   

The particular steps to be taken in the notebook are:  
1) [Import of the datasets](#import)  
2) [Detection of human faces](#Detect_Human)  
3) [Detection of dogs](#Detect_Dog)  
4) Setup & training of a convolutional neural network (CNN) from scratch for classification  
5) Training of an existing CNN for dog breed classification using transfer learning  
6) Setup & training of a new CNN for dog breed classification using transfer learning  
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
The algorithm decomposes the image using Haar Wavelets at first and then leads found feature to a cascade of classifiers.
This is also called the [Viola-Jones method](https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html).
Testing the algorithm on a dataset of 100 dog images and 100 human face images, shows that:
* 100% of the human face images where correctly detected as humans
* 11% of the dog images where falsely detected as humans  

One backdraw of the Viola-Jones method is that it always needs a completely visible human face for a correct detection.
To avoid this problem, an app using the algorithm could simply tell the user if the face on the image is not clearly visible.
Another idea could be, to train the underlying model with images of people who's faces are not completely visible
so the model is better able to detect faces just by finding features like eyes, mouths or ears.

## Detection of dogs <a name="Detect_Dog"></a>  
