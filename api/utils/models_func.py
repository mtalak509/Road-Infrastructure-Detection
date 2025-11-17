import torch
from ultralytics import YOLO
# import torchvision.transforms as T
import json
# import joblib
import os


# def load_classes():
#     '''
#         Returns RoadTraficClassification classes
#     '''
#     with open('api/utils/model_classes.json') as f:
#         labels = json.load(f)
#     return labels

# def get_class(i):
#     '''
#         Input int: class index.
#         Returns name of classified obj.
#     '''
#     labels = load_classes()
#     return(labels[i])

def load_model():
    '''
        Load YOLO8 model.
        Returns loaded model.
    '''
    model = YOLO("weights/best.pt")
    model.eval()
    return model


    

