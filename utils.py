
# Importing libraries
import os
import pathlib
import cv2
import numpy as np
import scipy as sc
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tensorflow.keras.preprocessing.image import load_img, img_to_array


def predict_object_detection(model, preprocess_input, image_path, image_size):
    """
    Function predict the boundary box for an input image.
    Here, following parameter should be passed while calling the function.
    model: trained CNN model
    preprocess_input: preprocess image based on the model
    image_path: image path to predict boudning box
    """
    unscaled_img = cv2.imread(image_path)
    height, width, _ = unscaled_img.shape
    scaled_img = cv2.resize(unscaled_img, (224, 224))
    
    # Applying the preprocessing on the image.
    feat_scaled_img = preprocess_input(np.array(scaled_img, dtype=np.float32))
    
    # Predict the Boundaty Box
    scaled_bbox = model.predict(x=np.array([feat_scaled_img]))[0]
    
    
    # Rescaling the predicted boundary box
    unscaled_bbox = np.array([scaled_bbox[0]*width/image_size, scaled_bbox[1]*height/image_size, scaled_bbox[2]*width/image_size, scaled_bbox[3]*height/image_size])
    
    # Image dimension
    pred_x0 = unscaled_bbox[0]
    pred_y0 = unscaled_bbox[1]
    pred_x1 = unscaled_bbox[2]
    pred_y1 = unscaled_bbox[3]

    # Displaying the boundary box of the image
    plt.imshow(unscaled_img)
    plt.gca().add_patch(patches.Rectangle((pred_x0, pred_y0), pred_x1 - pred_x0, pred_y1- pred_y0, fill=False, linewidth=2, edgecolor='r'))
    plt.axis("off")

def predict_classification(model, bbox, preprocess_input, image_path, label_encoder, image_size=(224, 224)):
    """
    Function to predict the classification of car images
    Here, following parameter should be passed while calling the function.
    model: trained CNN model
    preprocess_input: preprocess image based on the model
    image_path: image path to predict boudning box
    """

    xmin = bbox[0]
    ymin = bbox[1]
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    
    image = load_img(image_path)
    image_arr = img_to_array(image)

    image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
    image_arr = cv2.resize(image_arr,(image_size[0], image_size[1]))
    
    image_arr = image_arr/255.

    # Applying the preprocessing on the image.
    feat_scaled_img = preprocess_input(np.array(image_arr, dtype=np.float32))

    # Predict the class
    class_predict = model.predict(x=np.array([feat_scaled_img]))

    # Get argmax from the prediction
    class_predict_argmax = np.array([np.argmax(class_predict)])

    # Get Target class
    target_class = label_encoder.inverse_transform(class_predict_argmax)

    return target_class