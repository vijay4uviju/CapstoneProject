# Importing libraries
from base64 import encode
import pickle
import cv2
import pickle
import sys
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.image import imread

from data_loader import CustomDataGenFromDataFrame
from metrics import IoU

import tensorflow
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Flatten, Dense, BatchNormalization, 
                                     Conv2D, MaxPooling2D, Reshape, Dropout)
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input as vgg16_preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.optimizers import Adam, SGD
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class CarObjectDetection:
    """
    CarObjectDetection class contains the functinality of loading data, to training, to prediction.

    
    """
    
    def __init__(self):
        self.data_loaded = False
        self.data_preprocessed = False
        
    def load_data_directory(self, file_path):
        """
        Loading the data starts with entering the main project directory 
        where all the required data are stored. The main directory should 
        contain the following files and folders.
        1. Directory: Car Images = Main directory for car images with sub-directories
                      'Train Images' and 'Test Images'.
        2. Directory: Annotaion = Two CSV files 'Train Annotations.csv' and 'Test Annotation.csv'.
                      These files contain the class lables and bounding box co-ordinates for each images.
                      We will be using these two CSV files for training and testing part of the problem.
        3. File: Car names and make.csv = Car model name and make year of all car in 
        
        =======
        file_path: str: Windows or Linux path to the directory
        """
        self.proj_dir = pathlib.Path(file_path)
        if not self.proj_dir.exists():
            print("The path doesn't exist. Make sure the correct path is entered.")
            return
        
        # Getting data directory paths
        self._get_data_directories()
        
        # Getting dataframe paths
        self._load_annotation()
        self._load_car_name_make()
        
        # Loading dataframes to the memory
        self._load_dataframe()
        
        self.data_loaded = True
        print("Train and test datasets are loaded successfully.")
        print("The car images will loaded during the training.")
    
    def _get_data_directories(self):
        self.car_images = self.proj_dir.joinpath("Car Images")
        self.car_images_train = self.car_images.joinpath("Train Images")
        self.car_images_test = self.car_images.joinpath("Test Images")
    
    def _load_annotation(self):
        self.annotations_dir = self.proj_dir.joinpath("Annotations")
        self.train_annotations = self.annotations_dir.joinpath("Train Annotations.csv")
        self.test_annotations = self.annotations_dir.joinpath("Test Annotation.csv")
    
    def _load_car_name_make(self):
        self.car_names_make = self.proj_dir.joinpath("Car names and make.csv")
        
    def _load_dataframe(self):
        self.df_train = pd.read_csv(self.train_annotations)
        self.df_test = pd.read_csv(self.test_annotations)
        self.df_train_test = [self.df_train, self.df_test]
        
        self.df_car_names_make = pd.read_csv(self.car_names_make, header=None)
    
    def load_data_zip(self, zip_file_path):
        zip_path = pathlib.Path(zip_file_path)
        temp_dir = zip_path.parent
        pass
    
    def get_eda(self):
        if not self.data_loaded:
            print("Load the data first to the memory")
            return
        print(f"""
        BASIC EDA:
        1. Shape of training data:{self.df_train.shape}
        2. Shape of test data:{self.df_test.shape}

        We have another dataframe that contains the cars information. 
        The shape of car name and make year data: {self.df_car_names_make.shape}

        We have {self.df_car_names_make.shape[0]} target variables. 

        """)
    
    def _get_image_file_path(self, row, main_path):
        car_make_year = row["Car Name and Make year"]
        image_name = row["Image Name"]
        return str(main_path.joinpath(car_make_year).joinpath(image_name))
    
    def _get_dimension(self,img_path):
        return imread(img_path).shape
    
    def _scale_bbox(self, df, image_size=224):
        df["Dimension"] = df.apply(lambda row: self._get_dimension(row["Filepath"]), axis=1)
        
        df["Scale_X0"] =  df.apply(lambda row: row["Bounding Box: X0"]*image_size/row["Dimension"][1], axis=1)
        df["Scale_Y0"] =  df.apply(lambda row: row["Bounding Box: Y0"]*image_size/row["Dimension"][0], axis=1)
        df["Scale_X1"] =  df.apply(lambda row: row["Bounding Box: X1"]*image_size/row["Dimension"][1], axis=1)
        df["Scale_Y1"] =  df.apply(lambda row: row["Bounding Box: Y1"]*image_size/row["Dimension"][0], axis=1)
        return df
    
    def _pre_process(self, df, flag, image_size=224):
        """
        By default, the flag is selected as train. Please make sure to choose right flag to use for path of images.
        """
        rename_columns = {"Bounding Box coordinates": "Bounding Box: X0",
                          "Unnamed: 2": "Bounding Box: Y0",
                          "Unnamed: 3": "Bounding Box: X1",
                          "Unnamed: 4": "Bounding Box: Y1"}
        fix_car_name = {'Ram C/V Cargo Van Minivan 2012':'Ram C-V Cargo Van Minivan 2012'}
        
        car_train_test_path = {"train": self.car_images_train, 
                               "test": self.car_images_test}
        
        df["Car Name and Make year"] = df.apply(lambda row: self.df_car_names_make.iloc[row["Image class"]-1,0], axis=1)
        df = df.rename(columns=rename_columns)
        df['Car Name and Make year'] = df['Car Name and Make year'].replace(fix_car_name)
        df["Filepath"] =  df.apply(lambda row: self._get_image_file_path(row, car_train_test_path[flag]), axis=1)
        df = self._scale_bbox(df)
        return df
           
    def pre_process(self):
        if not self.data_loaded:
            print("Data is not loaded to the memory. Run `load_data_directory` function.")
            return
        
        self.df_train = self._pre_process(self.df_train, flag="train")
        self.df_test = self._pre_process(self.df_test, flag="test")
        print("""
        In pre-processing, following steps are took care:
        1. Checked the data for null values and drop the data if found any
        2. Fixed the error in one of the target variable 'Ram C/V Cargo Van Minivan 2012'. The target variable was not matching the directory of the images stored.
        3. Included new column "Filepath" to load images.
        4. Included four new columns  scaled the boudning boxes based standard image_size.          
        
        """)
        
    def get_training_data(self):
        return self.df_train
    
    def get_test_data(self):
        return self.df_test
    
    def get_bbox_data_generator(self):
        if not self.data_preprocessed:
            self.pre_process()
        self.bbox_image_train_data_generator = ImageDataGenerator(rescale=1/255, validation_split=0.25)
        self.bbox_train_iterator = self.bbox_image_train_data_generator.flow_from_dataframe(self.df_train,
                                                                                     x_col = "Filepath",
                                                                                     y_col = ["Scale_X0", "Scale_Y0", "Scale_X1", "Scale_Y1"],
                                                                                     color_mode="rgb",
                                                                                     target_size=(224,224),
                                                                                     batch_size=8,
                                                                                     seed=27,
                                                                                     class_mode="raw",
                                                                                     subset='training')
        self.bbox_val_iterator = self.bbox_image_train_data_generator.flow_from_dataframe(self.df_train,
                                                                                     x_col = "Filepath",
                                                                                     y_col = ["Scale_X0", "Scale_Y0", "Scale_X1", "Scale_Y1"],
                                                                                     color_mode="rgb",
                                                                                     target_size=(224,224),
                                                                                     batch_size=8,
                                                                                     seed=27,
                                                                                     class_mode="raw",
                                                                                     subset='validation')
        
        self.bbox_image_test_data_generator = ImageDataGenerator(rescale=1/255)
        self.bbox_test_iterator = self.bbox_image_test_data_generator.flow_from_dataframe(self.df_test,
                                                                                         x_col = "Filepath",
                                                                                         y_col = ["Scale_X0", "Scale_Y0", "Scale_X1", "Scale_Y1"],
                                                                                         color_mode="rgb",
                                                                                         target_size=(224,224),
                                                                                         batch_size=8,
                                                                                         seed=27,
                                                                                         class_mode="raw")
        print("""
        Image data is loaded to the memory using `ImageDataGenerator` with `flow_from_dataframe`.

        For training, we are splitting the data to 75:25 ration for training and validation.
        """)
        
    def object_detection_model(self, pre_trained_model, trainable=False):
        """
        A function to create a tensorflow model on pretrained model.
        =================

        Inputs:
        pretrained_model: Pass pre-trained keras.application model.
        trainable: Whether to train the pretrained layer or not. Default, False.
        """
        # Selecting pre-trained model
        model = pre_trained_model    
        # Freezeing already trained layers
        for layer in model.layers:
            layer.trainable = trainable
        # Adding new top layers
        x0 = model.layers[-1].output
        x1 = Conv2D(4, kernel_size=4)(x0)
        x1 = Conv2D(4, kernel_size=4, name="coords")(x1)
        x2 = Reshape((4,))(x1)
        return Model(inputs=model.input, outputs=x2)

    def training_bbox(self, pre_trained_model=VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3)), epochs=5, trainable=False):
        
        # Getting image data generator
        self.get_bbox_data_generator()
        
        callback_object = tensorflow.keras.callbacks.EarlyStopping(monitor='val_IoU', patience=5, min_delta=0.01)
        self.model_object_detection = self.object_detection_model(pre_trained_model=pre_trained_model)
        self.model_object_detection.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=0.001), metrics=[IoU])
        self.model_object_detection.fit(self.bbox_train_iterator, validation_data=self.bbox_val_iterator, epochs=epochs, verbose=2, callbacks=[callback_object])
        
        self.model_object_detection_losses = pd.DataFrame(self.model_object_detection.history.history)
        
    def loading_saved_bbox_model(self, model_path):
        """
        """
        self.model_object_detection = keras.models.load_model(model_path, custom_objects={'IoU':IoU})
    
    def saving_bbox_model(self, model_path):
        """
        """
        self.model_object_detection.save(model_path)
        print(f"The model saved sucessfully at {model_path}!")
    
    def evaluation_bbox(self):
        """
        Evaluation of the trained model using test data
        """
        losses, accuracy = self.model_object_detection.evaluate(self.bbox_test_iterator, verbose=0)
        print(f"The accuracy of the test dataset:{accuracy}")
        print(f"The loss of the test dataset:{losses}")
    
    def prediction_bbox(self, image, preprocess_input=mobilenet_preprocess_input, image_size=224, raw=False):
        """
        Function predict the boundary box for an input image.
        Here, following parameter should be passed while calling the function.
        model: trained CNN model
        preprocess_input: preprocess image based on the model
        image_path: image path to predict boudning box
        """
        if raw:
            unscaled_img =  np.array(image) 
        else:
            unscaled_img = cv2.imread(image)
        height, width, _ = unscaled_img.shape
        scaled_img = cv2.resize(unscaled_img, (image_size, image_size))

        # Applying the preprocessing on the image.
        feat_scaled_img = preprocess_input(np.array(scaled_img, dtype=np.float32))

        # Predict the Boundaty Box
        scaled_bbox = self.model_object_detection.predict(x=np.array([feat_scaled_img]))[0]


        # Rescaling the predicted boundary box
        unscaled_bbox = np.array([scaled_bbox[0]*width/image_size, scaled_bbox[1]*height/image_size, scaled_bbox[2]*width/image_size, scaled_bbox[3]*height/image_size])

        return unscaled_bbox

        """
        # Image dimension
        pred_x0 = unscaled_bbox[0]
        pred_y0 = unscaled_bbox[1]
        pred_x1 = unscaled_bbox[2]
        pred_y1 = unscaled_bbox[3]

        # Displaying the boundary box of the image
        plt.imshow(unscaled_img)
        plt.gca().add_patch(patches.Rectangle((pred_x0, pred_y0), pred_x1 - pred_x0, pred_y1- pred_y0, fill=False, linewidth=2, edgecolor='r'))
        plt.axis("off")
        plt.savefig('prediction.png', bbox_inches='tight', pad_inches=0)"""
    
    def _get_extra_features(self, df):
        df["BoundingBox"] = df.apply(lambda row: [row["Bounding Box: X0"], row["Bounding Box: Y0"], row["Bounding Box: X1"], row["Bounding Box: Y1"]], axis=1)
        
        self.target_encoder = LabelEncoder()
        self.target_encoder.fit(df["Car Name and Make year"].values)
        df["Target Class"] = self.target_encoder.transform(df["Car Name and Make year"].values)
        return df
    
    def get_classification_data_generator(self):
        self.df_train = self._get_extra_features(self.df_train)
        self.df_test = self._get_extra_features(self.df_test)

        df_train_classify = self.df_train.sample(frac = 0.75, random_state=27)
        df_train_val = self.df_train.drop(df_train_classify.index)

        self.classify_train_iterator = CustomDataGenFromDataFrame(df_train_classify,
                                                                    X_col="Filepath",
                                                                    Bbox = "BoundingBox",
                                                                    y_col= "Target Class",
                                                                    batch_size=8, input_size=(224,224))
        
        self.classify_val_iterator = CustomDataGenFromDataFrame(df_train_val,
                                                                    X_col="Filepath",
                                                                    Bbox = "BoundingBox",
                                                                    y_col= "Target Class",
                                                                    batch_size=8, input_size=(224,224))
        
        self.classify_test_iterator = CustomDataGenFromDataFrame(self.df_test,
                                                                    X_col="Filepath",
                                                                    Bbox = "BoundingBox",
                                                                    y_col= "Target Class",
                                                                    batch_size=8, input_size=(224,224))
        
        print("""
        Image data is loaded to the memory using `CustomDataGenFromDataFrame`. 
        During this process, we added additional columns to both the train and test datasets.

        For training, we are splitting the data to 75:25 ration for training and validation.
        """)
        
    def classification_model(self, pre_trained_model, trainable=False):
        """
        A function to create a tensorflow model on pretrained model for image classification.
        =================
        
        Inputs:
        pretrained_model: Pass pre-trained model from tensorflow.keras.application model.
        trainable: Whether to train the pretrained layer or not. Default, False.
        """
        # Selecting pre-trained model
        model = pre_trained_model    
        
        # Freezeing already trained layers
        for layer in model.layers:
            layer.trainable = trainable

        # Adding new top layers
        x0 = model.layers[-1].output
        x1 = Flatten()(x0)
        
        x1 = BatchNormalization(epsilon=0.001)(x1)
        
        x1 = Dense(512, activation="relu")(x1)
        x1 = Dropout(0.5)(x1)
        
        x1 = BatchNormalization(epsilon=0.001)(x1)
        
        x1 = Dense(256, activation="relu")(x1)
        x1 = Dropout(0.5)(x1)
        #x1 = BatchNormalization()(x1)
        
        #x1 = Dense(256, activation="relu")(x1)
        #x1 = BatchNormalization(x1)
        #x1 = Dropout(0.5)(x1)
        
        x2 = Dense(196, activation="softmax")(x1)
        
        
        return Model(inputs=model.input, outputs=x2)
    
    def training_classify(self, pre_trained_model=MobileNet(include_top=False, weights="imagenet", input_shape=(224,224,3), alpha=1.0), epochs=5, trainable=False):
        
        # Getting image data generator
        self.get_classification_data_generator()

        callback_cnn = tensorflow.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01)
        self.model_classify = self.classification_model(pre_trained_model=pre_trained_model)
        self.model_classify.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001), metrics=["accuracy"])
        self.model_classify.fit(self.classify_train_iterator, validation_data=self.classify_val_iterator, epochs=epochs, verbose=0, callbacks=[callback_cnn])
        
        self.model_classify_losses = pd.DataFrame(self.model_classify.history.history)
    
    def loading_saved_classify_model(self, model_path: pathlib.Path):
        """
        """
        self.model_classify = keras.models.load_model(model_path)
    
    def saving_classify_model(self, model_path: pathlib.Path):
        """
        """
        encoder_path = model_path.parent.joinpath("target_encoder.pkl")
        self.model_classify.save(model_path)
        with open("target_encoder.pkl", "wb") as file:
            pickle.dump(encoder_path, file)
        print(f"The models saved sucessfully in {model_path}")
    
    def evaluation_classify(self):
        """
        Evaluation of the trained model using test data
        """
        losses, accuracy = self.model_classify.evaluate(self.classify_test_iterator, verbose=0)
        print(f"The accuracy of the test dataset:{accuracy}")
        print(f"The loss of the test dataset:{losses}")
    
    def load_label_encoder(self, filepath):
        with open(filepath, "rb") as file:
            self.target_encoder = pickle.load(file)
            

    def predict_classify(self, image, bbox, preprocess_input=mobilenet_preprocess_input, image_size=(224, 224), raw=False):
        """
        Function to predict the classification of car images
        Here, following parameter should be passed while calling the function.
        preprocess_input: preprocess image based on the model
        image_path: image path to predict boudning box
        """

        xmin = int(bbox[0])
        ymin = int(bbox[1])
        w = int(bbox[2] - bbox[0])
        h = int(bbox[3] - bbox[1])

        if raw:
            image_arr = np.array(image)
        else:
            image_arr = cv2.imread(image)

        image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        image_arr = cv2.resize(image_arr,(image_size[0], image_size[1]))

        image_arr = image_arr/255.

        # Applying the preprocessing on the image.
        feat_scaled_img = preprocess_input(np.array(image_arr, dtype=np.float32))

        # Predict the class
        class_predict = self.model_classify.predict(x=np.array([feat_scaled_img]))

        # Get argmax from the prediction
        class_predict_argmax = np.array([np.argmax(class_predict)])

        # Get Target class
        target_class = self.target_encoder.inverse_transform(class_predict_argmax)[0]

        return target_class