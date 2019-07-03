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
from fastai.vision.data import ImageDataBunch
from fastai.vision.learner import cnn_learner
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image


def loadModel():

    # Retrieves trained model from SortingTraining Directory
    path = os.getcwd() + "/data"

    # path = "/Users/zeke/PycharmProjects/15112_TP_S19/SortingTraining/data"
    data = ImageDataBunch.from_folder(path, train="train", valid="valid")

    # Loads trained model, exports for inference, and loads inference model
    learn = cnn_learner(data, models.resnet34).load('recyc_model')
    learn.export()
    learn = load_learner(path)

    return learn


def connectCaptureDevice():

    # Connects to webcam; if fails, attempts again
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
        quit()

    return webcam


def prediction(learn, input):

    # Forward propagates through nn to classify image
    pred, idx, probs = learn.predict(Image(pil2tensor(input, np.float32).div_(255)))

    # If accurate prediction can't be made, chooses trash
    if max(probs) < 0.5:
        pred = "Trash"
        return pred

    # Sorts classification of waste type into disposal categories
    if pred == "paper" or pred == "cardboard":
        pred = "Paper"
    elif pred == "glass" or pred == "metal":
        pred = "Glass/Metal"
    elif pred == "plastic":
        pred = "Plastic"
    else:
        pred = "Trash"

    return pred


def main():

    # Loads image classifying model and retrieves webcam capture
    learn = loadModel()
    webcam = connectCaptureDevice()

    while True:
        # Capture frame-by-frame
        ret, frame = webcam.read()

        # Display the resulting frame
        cv.imshow('Feed', frame)
        input = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        category = prediction(learn, input)

        # Exit Standby
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyWindow('Feed')
            webcam.release()
            break

        """elif cv.waitKey(1) & 0xFF == ord(' '):
            print("Command received!")
            __, still = webcam.read()
            cv.imshow('still', still)"""

    # When everything done, end script


if __name__ == '__main__':
    main()



