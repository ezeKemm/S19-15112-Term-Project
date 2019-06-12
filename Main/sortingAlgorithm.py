from __future__ import print_function, division

import cv2
import numpy as np
import time
import collections
import math
import sys


def main():

    MAX_RETRIES = 10
    for i in range(MAX_RETRIES):
        try:
            webcam = cv2.VideoCapture(0)
        except:
            print("Failed to connect to webcam, retrying...")
            continue
        else:
            break
    else:
        print("Failed to successfully connect to camera after 10 tries")



    pass