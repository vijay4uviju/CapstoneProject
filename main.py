# Importing libraries
import random
import os
import re
import cv2
import pathlib
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

# Configuration
sns.set(color_codes=True) 
sns.set_theme(style="whitegrid")

# Scikit-learn packages for accuracy check
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.model_selection import train_test_split

# Python Imaging Library
import cv2
import skimage.io as io
from PIL import Image
from skimage import color
from skimage.transform import rescale, resize
from skimage.color import rgb2gray,rgba2rgb

# Importing tensorflow packages
import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Input, Flatten, Dense, BatchNormalization, 
                                     Conv2D, MaxPooling2D, Reshape,
                                     Conv3D, MaxPooling3D, Dropout)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input as inception_preprocess_input
from tensorflow.keras.applications.resnet_v2 import ResNet101V2, ResNet152V2


from classifier import CarObjectDetection
from metrics import IoU
from data_loader import CustomDataGenFromDataFrame


# Project data directory
data_dir = pathlib.Path("C:\\Users\\ELECTROBOT\\Downloads\\GL-Capstone-CV-2")


# Intializing the car object detection
car_object_detection = CarObjectDetection()


# Loading the data directory
print("\nLoading the data directory..")
car_object_detection.load_data_directory(data_dir)


# Printing training data
print("\nDisplaying Training data")
print(car_object_detection.get_training_data().head())


# Printing testing data
print("\nDisplaying Test data")
print(car_object_detection.get_test_data().head())


# Printing testing data
print("\nEDA")
car_object_detection.get_eda()


# Pre-processing the data
print("\nData Pre-processing")
car_object_detection.pre_process()


# Training
print("\nTraining for BBOX")
car_object_detection.training_bbox(pre_trained_model=MobileNet(include_top=False, weights="imagenet", input_shape=(224,224,3), alpha=1.0))


# Saving the model
print("\nSaving BBOX model")
car_object_detection.saving_bbox_model("saved_models/bbox_mobilenet_v1.h5")


# Evaluation the test data
print("\nEvaluating for test data for BBOX")
car_object_detection.evaluation_bbox()


"""
# Loading the model
print("\nLoading the model for BBOX")
car_object_detection.loading_saved_bbox_model("saved_models/model_bbox_mobilenet_v1.h5")

print(car_object_detection.model_object_detection.summary())
"""

# Prediction for BBOX.
print("\nPredicting for BBOX")
print(car_object_detection.get_test_data().head())
print(car_object_detection.get_test_data().columns)

sample_image = car_object_detection.get_test_data().iloc[2,7]
car_object_detection.prediction_bbox(sample_image)


# Training classification
print("\nTraining for classification")
car_object_detection.training_classify(pre_trained_model=MobileNet(include_top=False, weights="imagenet", input_shape=(224,224,3), alpha=1.0))


# Saving the classification model
print("\nSaving classification model")
car_object_detection.saving_classify_model("saved_models/classify_mobilenet_v1.h5")


# Evaluation the test data for classification
print("\nEvaluating for test data for classification")
car_object_detection.evaluation_classify()




