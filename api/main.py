import logging
import random
from contextlib import asynccontextmanager

import PIL
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, UploadFile
from pydantic import BaseModel
from utils.models_func import load_model

logger = logging.getLogger('uvicorn.info')

class ImageResponse(BaseModel):
    class_indices: list  # class index
    confidences: list # model confidence
    class_names: list # class name

yolo_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Контекстный менеджер для инициализации и завершения работы FastAPI приложения.
    Загружает модели машинного обучения при запуске приложения и удаляет их после завершения.
    """

    global yolo_model
    yolo_model = load_model()
    logger.info('yolo model loaded')
    yield
    del yolo_model

app = FastAPI(lifespan=lifespan)

@app.get('/')
def return_info():
    """
    Возвращает приветственное сообщение при обращении к корневому маршруту API.
    """
    return 'Hello FastAPI!'

@app.post('/clf_image')
def classify_image(file: UploadFile):
    '''
        Endpoin for image classification.
        Predict class of uploaded fife, returns class name, class index
    '''
    image = PIL.Image.open(file.file)
    logger.info(f'uploaded image : {image}')
    with torch.inference_mode():
        results = yolo_model(image)
        result = results[0]

        # Получаем все данные разом
        boxes = result.boxes

        # Извлекаем классы, уверенность и координаты
        class_indices = boxes.cls.int().tolist()    #list
        confidences = boxes.conf.tolist()   #list
        class_names = [yolo_model.names[i] for i in class_indices]  #lsit

    response = ImageResponse(class_indices=class_indices,
                             confidences=confidences,
                             class_names=class_names)
    return response





