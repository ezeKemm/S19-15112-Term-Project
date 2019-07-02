from __future__ import print_function, division

import cv2 as cv
import numpy as np
import time
import collections
import math
import sys
import os

from pathlib import Path
from fastai.vision import *
from fastai.vision.data import *
from fastai.vision.learner import create_cnn
from fastai.vision import models
from fastai.vision.image import pil2tensor,Image


def loadTrainedModel():
    MODEL_PB = ""
    MODEL_CONFIG = ""

    retrievedModel = cv.dnn.readNetFromTensorflow(MODEL_PB, MODEL_CONFIG)

    return retrievedModel


def main():
    # Retrieves Trained model from SortingTraining Directory
    path = "/Users/zeke/PycharmProjects/15112_TP_S19/SortingTraining/data"
    tfms = get_transforms(do_flip=False)
    data = ImageDataBunch.from_folder(path, train="train", valid="valid")

    learn = cnn_learner(data, models.resnet34).load('recyc_model')
    learn.export()
    learn = load_learner(path)

    # Connects to webcam, If fails attempts again
    MAX_RETRIES = 10
    for i in range(MAX_RETRIES):
        try:
            webcam = cv.VideoCapture(0)
        except:
            print("Failed to connect to webcam, retrying...")
            continue
        else:
            break
    else:
        print("Failed to successfully connect to camera after 10 tries")

    while True:
        # Capture frame-by-frame
        ret, frame = webcam.read()

        # Display the resulting frame
        cv.imshow('Feed', frame)
        imgInput = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pred, idx, probs = learn.predict(Image(pil2tensor(imgInput, np.float32).div_(255)))
        print(pred, probs)

        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyWindow('Feed')
            webcam.release()
            break

        """elif cv.waitKey(1) & 0xFF == ord(' '):
            print("Command received!")
            __, still = webcam.read()
            cv.imshow('still', still)"""

    # When everything done, release the capture



if __name__ == '__main__':
    main()



