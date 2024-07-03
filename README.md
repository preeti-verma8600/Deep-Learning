# Deep Learning 

This repository contains three Jupyter notebooks focused on different deep learning tasks: image segmentation, object detection, and image captioning.

## Table of Contents

- [Notebook 1: Image Classification](#task-1-image-classification)
- [Notebook 2: Object Detection](#task-2-object-detection)
- [Notebook 3: Image Captioning](#task-3-image-captioning)


## Task 1: Image classification

The goal of this task is to compare the performance of a given pre-trained model and a modified version of it, using a given dataset (https://www.kaggle.com/code/turusanasak/cats-and-dogs). The datasets contain images as input and class labels as target, thus task is to solve a Supervised Machine Learning Classification problem.

**Dataset**:
- Dataset has two classes: Cat and Dog
- The dataset should be split into train, validation and test sets. Some datasets already have these splits, otherwise, you can split the training set into validation and test sets.
- Use the test set to do inference with the pre-trained model. Calculate the accuracy of the pre-trained model.

**Evaluation**:  Use the test set to do inference with the newly modified/trained model. Calculate the accuracy.

[Notebook for Image Classification](Image_Classification.ipynb)

## Task 2: Object Detection

This assignment focuses on fine-tuning a pre-trained object detection model (Yolov8n) with a new object category. 
- Curated new dataset bagpack for detection task using Roboflow: [Dataset](https://universe.roboflow.com/preetiverma/bagpack-dataset).
- Fine-tune the YOLOv8 model.

**Dataset**:
- Annotate 545 images.

**Evaluation**: Performance measured using mAP (mean Average Precision).

[Notebook for Object Detection](Detection.ipynb)

## Task 3: Image Captioning

This assignment involves creating an image captioning model. Given an image, the model predicts a caption that describes the image. The model uses an encoder-decoder architecture:
- **Encoder**: Vision-based CNN.
- **Decoder**: Sequence-based model (RNN, LSTM, Transformer, etc.).

**Dataset**:
- Training and validation: [Flickr8k dataset](https://www.kaggle.com/datasets/sayanf/flickr8k/)

**Evaluation**: BLEU score.

[Notebook for Image Captioning](Image_Captioning.ipynb)


## Getting Started

To get started with these notebooks, clone this repository and navigate to the notebooks directory.

```sh
git clone https://github.com/preeti-verma8600/Deep-Learning.git
