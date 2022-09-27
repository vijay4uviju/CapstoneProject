# Importing general libraries
import cv2
import random
import math
import pathlib
import time
import pandas as pd
import numpy as np
from io import StringIO

# Importing streamlit related libraries
import tkinter as tk
from tkinter import filedialog
import streamlit as st
from contextlib import contextmanager, redirect_stdout

# Python Imaging Library
from PIL import Image, ImageFont, ImageDraw, ImageEnhance


# Importing tensorflow related modules
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input as vgg19_preprocess_input

# Importing User defined modules
from classifier import CarObjectDetection


# Defining constant variables
TEMP_DIR = pathlib.Path(__file__).parent.joinpath("temp")

RAND_INT = str(random.randint(111,999))

SAVED_MODEL_PATH = pathlib.Path(__file__).parent.joinpath("saved_models")
SAVED_MODELS = {}
for model in SAVED_MODEL_PATH.iterdir():
    SAVED_MODELS[model.name]={
        "bbox_model": model.joinpath("bbox.h5"),
        "classification_model": model.joinpath("classification.h5"),
        "target_encoder": model.joinpath("target_encoder.pkl")
    }

PRE_TRAINED_MODELS = {}
PRE_TRAINED_MODELS["MobileNet"] = {
    "pre_trained_model": MobileNet(include_top=False, weights="imagenet", input_shape=(224,224,3), alpha=1.0),
    "pre_process_input": mobilenet_preprocess_input
}
PRE_TRAINED_MODELS["VGG16"] = {
    "pre_trained_model": VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3)),
    "pre_process_input": vgg16_preprocess_input
}
PRE_TRAINED_MODELS["VGG19"] = {
    "pre_trained_model": VGG19(include_top=False, weights="imagenet", input_shape=(224,224,3)),
    "pre_process_input": vgg19_preprocess_input
}

# Sett up tkinter to load folder path
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)


@contextmanager
def st_capture(output_func):
    """
    Function to display the stdout on streamlit.
    """
    with StringIO() as stdout, redirect_stdout(stdout):
        old_write = stdout.write

        def new_write(string):
            ret = old_write(string)
            output_func(stdout.getvalue())
            return ret
        
        stdout.write = new_write
        yield   


@st.cache
def loadImage(img_file):
    """
    Cached function to load image on streamlit
    """
    img = Image.open(img_file)
    return img

@st.cache
def add_bbox(img_file, bbox):
    """
    Cached function to add rectangle box to the identified object and return as an image for streamlit.
    """
    img = Image.open(img_file)
    bbox_width = math.ceil(max(img.size)*0.01/2)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle(bbox, outline="red", width=bbox_width)
    return img


# Sidebar part of the UI to do the prediction using saved model.
st.sidebar.subheader("Object Detection using Saved Model")

option = st.sidebar.selectbox(
    "Please select a saved model and upload an image for prediction.",
    SAVED_MODELS.keys())

image_file = st.sidebar.file_uploader("", type=["png", "jpeg", "jpg"])

if image_file is not None:

    file_metadata = {"FileName": image_file.name, "FileType": image_file.type}
    new_prediction = CarObjectDetection()
    new_prediction.loading_saved_bbox_model(SAVED_MODELS[option]["bbox_model"])
    new_prediction.loading_saved_classify_model(SAVED_MODELS[option]["classification_model"])
    new_prediction.load_label_encoder(SAVED_MODELS[option]["target_encoder"])

    bbox_prediction = new_prediction.prediction_bbox(loadImage(image_file), raw=True)
    class_prediction = new_prediction.predict_classify(loadImage(image_file),bbox_prediction.tolist(), raw=True)

    file_metadata = {
        "fileName": image_file.name,
        "fileType": image_file.type,
        "boundary_box": bbox_prediction.tolist(),
        "car_name_make_year": class_prediction
        }

    st.sidebar.write(file_metadata)
    st.sidebar.image(add_bbox(image_file, bbox_prediction.tolist()))


# Main body of UI where the training and validation and saving the model happens.

st.title("Image Classification and Object Detection")

with open("doc.md", "r") as intro:
    st.markdown(intro.read())


model_selection = st.selectbox("Please select a pre-trained model to use for training.", PRE_TRAINED_MODELS.keys())
project_name = st.text_input('Select the project name to save the model.', model_selection.lower())
total_epochs = st.number_input('Enter the number of epochs for training', value=5, format="%d")

st.write('By clicling the below button, the project data directory will be selected and analysis will begin.')

with st.container(), st.spinner():
    clicked = st.button('Run')

    if clicked:
        selected_data_dir = st.text_input('Selected folder:', filedialog.askdirectory(master=root))

        if not pathlib.Path(selected_data_dir).exists or selected_data_dir == "":
                st.write("Unable to locate the data path.")      
        else:
            st.write("Intializing the Object Detection")
            car_object_detection = CarObjectDetection()

            st.write("Loading the data directory..")
            data_loading = st.empty()
            with st_capture(data_loading.text):
                car_object_detection.load_data_directory(selected_data_dir)
            
            st.write("Displaying Training data")
            st.dataframe(car_object_detection.get_training_data().head())
                        
            st.write("Displaying Test data")
            st.dataframe(car_object_detection.get_test_data().head())

            # Initialization
            if 'car_object' not in st.session_state:
                st.session_state['car_object'] = car_object_detection
            
            st.write("Performing data pre-processing")
            pre_process = st.empty()
            with st_capture(pre_process.text):
                st.cache(car_object_detection.pre_process())

            st.write("Basic Data Analysis")
            basic_eda = st.empty()
            with st_capture(basic_eda.text):
                car_object_detection.get_eda()
            
            st.write("Display Sample Images")

            st.write("Training and validating for bounding box model.")
            bbox_training = st.empty()
            with st_capture(bbox_training.code):
                st.cache(car_object_detection.training_bbox(pre_trained_model=PRE_TRAINED_MODELS[model_selection]["pre_trained_model"], epochs=total_epochs))
            
            bbox_eval = st.empty()
            with st_capture(bbox_eval.code):
                car_object_detection.evaluation_bbox()

            st.write("Saving the trained bbox model.")
            bbox_model_save = st.empty()
            with st_capture(bbox_model_save.code):
                car_object_detection.saving_bbox_model(SAVED_MODEL_PATH.joinpath(project_name+RAND_INT).joinpath("bbox.h5"))
            
            st.write("Training and validating for classification model.")
            class_training = st.empty()
            with st_capture(class_training.code):
                st.cache(car_object_detection.training_classify(pre_trained_model=PRE_TRAINED_MODELS[model_selection]["pre_trained_model"], epochs=total_epochs))
            
            class_eval = st.empty()
            with st_capture(class_eval.code):
                car_object_detection.evaluation_classify()

            st.write("Saving the trained classification model.")
            class_model_save = st.empty()
            with st_capture(class_model_save.code):
                car_object_detection.saving_classify_model(SAVED_MODEL_PATH.joinpath(project_name+RAND_INT).joinpath("classification.h5"))