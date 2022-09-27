import pathlib
from classifier import CarObjectDetection


SAVED_MODEL_PATH = pathlib.Path(__file__).parent.joinpath("saved_models")
#print(SAVED_MODEL_PATH, SAVED_MODEL_PATH.is_dir())

SAVED_MODELS = {}
for model in SAVED_MODEL_PATH.iterdir():
    SAVED_MODELS[model.name]={
        "bbox_model": model.joinpath("bbox.h5"),
        "classification_model": model.joinpath("classification.h5")
    }

#print(SAVED_MODELS)

option = "mobilenet_1"

new_prediction = CarObjectDetection()
new_prediction.loading_saved_bbox_model(SAVED_MODELS[option]["bbox_model"])
new_prediction.loading_saved_classify_model(SAVED_MODELS[option]["classification_model"])

bbox_prediction = new_prediction.prediction_bbox("test_images/00534.jpg")

print(bbox_prediction)

