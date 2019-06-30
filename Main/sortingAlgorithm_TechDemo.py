from __future__ import print_function, division

import cv2 as cv
import numpy as np
import time
import collections
import math
import sys
import os

import click
import cv2
import imutils
from pathlib import Path
from fastai.vision.data import ImageItemList
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

    path = Path(os.getcwd()) / "data"
    data = (
            ImageItemList.from_folder(path, csv_name="labels.csv")
            .no_split()
            .label_from_df(label_delim=" ")
            .transform(None, size=128)
            .databunch(no_check=True)
            .normalize()
    )

    learn = create_cnn(data, models.resnet34, pretrained=False)
    learn.load("recyc_model")

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
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
        elif cv.waitKey(1) & 0xFF == ord('e'):
            still = webcam.read()
            webcam.release()
            cv.imshow('still', still)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()

    # When everything done, release the capture
    webcam.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()



