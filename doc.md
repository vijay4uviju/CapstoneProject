**Domain**: Automotive. Surveillance.

**ContextT**:
Computer vision can be used to automate supervision and generate action appropriate action trigger if the event is predicted from the image of interest. For example a car moving on the road can be easily identified by a camera as make of the car, type, colour, number plates etc.

**Objective**: The objective of the project is to design a deep learning model to identify the car model.

**Data Descriptor**: The dataset contains $16,185$ images of $196$ classes of cars. The data is split into $8,144$ training images and $8,041$ testing images, where each class has been split roughly in a 50-50 split. 
Classes are typically at the level of Make, Model, Year, e.g. $2012$ Tesla Model S or $2012$ BMW M3 coupe.

**Analysis Plan**: We will start the training by selecting a pre-trained model for transfer learning. We will do the training for boundary box prediction and car class prediction separetely. By default, the system will run the 5 epochs and it can be increased. A early stopping callback also added to the training. Hence, the training will stop once the validation loss started increasing.


