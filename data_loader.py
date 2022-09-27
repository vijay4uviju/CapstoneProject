# Importing libraries
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.image import resize, transpose, flip_up_down, flip_left_right

class CustomDataGenFromDataFrame(Sequence):
    """
    Custom image data generator from the dataframe.
    
    ===============================================
    df: The dataframe that contains the image file path, bounding box, and target label.
    X_col: string: path
    Bbox: List[string]: Bbox Columns in the format [X0, Y0, X1, Y1]
    y_col: string: Target Column name
    """
    def __init__(self, df, X_col, Bbox, y_col, batch_size, input_size=(224, 224, 3), shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.Bbox = Bbox
        self.y_col = y_col
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle = shuffle
        
        self.image_aug = flip_left_right 
        
        self.n = len(self.df)
        self.n_target = df[y_col].nunique()
    
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, index):
        batches = self.df[index * self.batch_size:(index + 1) * self.batch_size]
        X, y = self.__get_data(batches)        
        return X, y
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __get_input(self, path, bbox, target_size):
        xmin = bbox[0]
        ymin = bbox[1]
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        image = load_img(path)
        image_arr = img_to_array(image)

        image_arr = image_arr[ymin:ymin+h, xmin:xmin+w]
        image_arr = resize(image_arr,(target_size[0], target_size[1])).numpy()
        
        image_arr = self.image_aug(image_arr)
        
        return image_arr/255.
    
    def __get_output(self, label, num_classes):
        return to_categorical(label, num_classes=num_classes)
    
    def __get_data(self, batches):
        """
        Generates data containing batch_size samples
        """
        path_batch = batches[self.X_col]
        bbox_batch = batches[self.Bbox]
        target_batch = batches[self.y_col]
        
        X_batch = np.asarray([self.__get_input(x1, x2, self.input_size) for x1, x2 in zip(path_batch, bbox_batch)])
        y_batch = np.asarray([self.__get_output(y, self.n_target) for y in target_batch])

        return X_batch, y_batch