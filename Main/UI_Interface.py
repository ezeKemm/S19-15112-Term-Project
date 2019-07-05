from __future__ import print_function, division

from tkinter import *
from PIL import Image, ImageTk
import time
import cv2 as cv
from pathlib import Path
import os

# This explicitly sets mpl back end as TkAgg
# to correct an error with Tkinter
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from fastai.vision import *
from fastai.vision.data import ImageDataBunch
from fastai.vision.learner import cnn_learner
from fastai.vision import models
from fastai.vision.image import pil2tensor, Image


# TODO -- Test splitting file for functionality
####################################
# Initializes
####################################


def load_model():

    # Retrieves trained model from SortingTraining Directory
    # Currently an error occurs if this path isn't explicitly given
    path = "/Users/zeke/PycharmProjects/15112_TP_S19/SortingTraining/data"
    data = ImageDataBunch.from_folder(path, train="train", valid="valid")

    # Loads trained model, exports for inference, and loads inference model
    learn = cnn_learner(data, models.resnet34).load('recyc_model')
    learn.export()
    learn = load_learner(path)

    return learn


# Connects to webcam; if fails, attempts again
def connect_capture_device():

    MAX_RETRIES = 10

    for i in range(MAX_RETRIES):
        try:
            webcam = cv.VideoCapture(0)

            # Sets to 720p resolution
            webcam.set(3, 1280)
            webcam.set(4, 1080)
        except:
            print("Failed to connect to webcam, retrying...")
            continue
        else:
            break
    else:
        print("Failed to successfully connect to camera after 10 tries")
        quit()

    return webcam


# Creates directory to store data
def load_folder():
    path = "./history"
    access = 0o755
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            return path
    except OSError:
        print("Couldn't create directory")


# Starts program, loading model and capture device
def main():
    global learn, webcam, classes, hist_path

    classes = ["Paper/Cardboard", "Plastic", "Glass/Metal", "Trash"]
    # Loads image classifying model and retrieves webcam capture
    learn = load_model()
    webcam = connect_capture_device()
    hist_path = load_folder()

    print(hist_path)
    print("Model loaded! Camera connected!")
    run(500, 400)


####################################
####################################

# Classifies image using machine learning,
# determines where item should be disposed of
def prediction(input, data):

    # Forward propagates through nn to classify image
    pred, idx, probs = learn.predict(Image(pil2tensor(input, np.float32).div_(255)))
    print(pred, probs, float(probs[idx]))

    # If accurate prediction can't be made, chooses trash
    if max(probs) < 0.5:
        data.predic = "Trash"
        data.confid = "0"
    else:
        data.predic = str(pred)
        data.confid = round(float(probs[idx]), 2)

    # Sorts classification of waste type into bin categories
    if str(pred) == "paper" or str(pred) == "cardboard":
        bin = 0
    elif str(pred) == "plastic":
        bin = 1
    elif str(pred) == "glass" or str(pred) == "metal":
        bin = 2
    else:
        bin = 3

    data.bin = classes[bin]
    data.button_fill[bin] = "light green"   # Highlights chosen bin category


# Captures webcam output, converts to be displayed on UI
def take_shot(data):
    data.counter += 1
    __, image = webcam.read()
    prediction(image, data)
    if data.counter > 0:    # Save image
        cv.imwrite(f"./history/capture{data.counter}.jpeg", image)

    # Conversions from OpenCV to Tkinter format using PIL, resized for UI
    resize_perc = 0.30
    width = int(image.shape[1] * resize_perc)
    height = int(image.shape[0] * resize_perc)
    image = cv.resize(image, (width, height), interpolation=cv.INTER_AREA)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = ImageTk.PhotoImage(image)
    return image


# IGNORE -- For testing purposes
def stream_capture(webcam, learn):

    while True:
        # Capture frame-by-frame
        ret, image = webcam.read()

        # Display the resulting frame
        cv.imshow('Feed', image)
        input = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        category = prediction(learn, input)

        # Exit Standby
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyWindow('Feed')
            webcam.release()
            break


# IGNORE -- Not operational
# TODO -- Fix timer so blink is always 500ms
def blink(data, idx):
    start = time.time()
    data.button_fill[idx] = "red"
    seconds = 3
    data.timer = 0
    while seconds > 0:
        timerFired(data)
        print(data.timer)
        if data.timer % 10 == 0:
           seconds -= 1

    data.button_fill[idx] = "white"
    end = time.time()
    totTime = end - start
    print(f"Time: {totTime}")
    return


def error_counter(correct, data):
    log = open("./history/prediction_log.txt", "a")

    if data.bin == data.button_labels[correct]:  # ML prediction correct
        data.button_fill[correct] = "blue"
        result = True
        prediction_result = f"Predicted {data.predic} in {data.bin} ... Correct: True"
    elif data.bin != data.button_labels[correct]:  # ML prediction incorrect
        data.button_fill[correct] = "red"
        result = False
        correct = data.button_labels[correct]
        prediction_result = f"Predicted {data.predic} in {data.bin} ... Correct: False -> {correct}"

    entry = f"#{data.counter} -- capture{data.counter}.jpeg -- " + prediction_result + "\n"
    print(entry)


    log.write(entry)
    log.close()


# Determines if click is in the buttons' boundaries
def check_cursor(x, y, data):

    height = data.height * data.bcell_height

    for button in range(data.num_buttons):
        start_x = button * data.bcell_width + data.x_margin
        end_x = button * data.bcell_width + data.bcell_width - data.x_margin
        start_y = data.height - height + data.y_margin
        end_y = data.height - data.y_margin

        if start_x <= x <= end_x and start_y <= y <= end_y:
            error_counter(button, data)
            # print(f"Click detected at Button {button+1}") # DEBUG



def init(data):

    # For the buttons
    data.num_buttons = len(classes)
    data.bcell_width = data.width // data.num_buttons
    data.bcell_height = 0.20    # represented as a fraction of total height
    data.x_margin = 5
    data.y_margin = 10
    data.button_fill = [ "white" for i in range(data.num_buttons) ]
    data.button_labels = classes

    data.counter = -1
    data.image = take_shot(data)
    data.predic = None
    data.confid = 0
    data.bin = None

    # No given prediction so no selection needed -- Slightly broken
    data.clicked = True
    data.checked = True


# TODO -- Add user prompts for else condition
def mousePressed(event, data):
    # use event.x and event.y
    if not data.clicked:
        data.clicked = True
        check_cursor(event.x, event.y, data)


def keyPressed(event, data):
    # use event.char and event.keysym
    # Captures image from device on press
    if data.checked:
        if event.keysym == "e":
            data.clicked = False
            data.image = take_shot(data)


def timerFired(data):
    # Reverts fill back -- Serves as click animation
    for button in range(data.num_buttons):
        if data.button_fill[button] == "red":
            data.button_fill[button] = "white"
        elif data.button_fill[button] == "blue":
            data.button_fill[button] = "light green"


def draw_buttons(canvas, data):
    height = data.height * data.bcell_height

    # Draw Buttons
    for button in range(data.num_buttons):
        start_x = button * data.bcell_width + data.x_margin
        end_x = button * data.bcell_width + data.bcell_width - data.x_margin
        start_y = data.height - height + data.y_margin
        end_y = data.height - data.y_margin

        # TODO -- Correct that clicking out of bounds still works
        if button == 0:
            start_x = button * data.bcell_width + (data.x_margin * 2)
        elif button == 3:
            end_x = button * data.bcell_width + data.bcell_width - (data.x_margin * 2)

        canvas.create_rectangle(start_x, start_y, end_x, end_y,
                                fill=data.button_fill[button])

        # Adds text labels to buttons
        txt_x = (end_x - start_x) // 2 + start_x
        txt_y = (height // 2) + (data.height - height) - data.y_margin  # Sloppy
        # TODO -- Try to find simpler calculation
        canvas.create_text(txt_x, txt_y,
                           text=data.button_labels[button],
                           anchor=N,
                           font=f"Sans {int(data.width * .025)} bold")


def redrawAll(canvas, data):

    # Draw panel background
    height = data.height * data.bcell_height
    canvas.create_rectangle(0, 0, data.width, data.height - height, fill="gray")
    canvas.create_rectangle(0, data.height - height, data.width, data.height, fill="black")

    draw_buttons(canvas, data)

    # Displays images from capture device
    im_height, im_width = data.image.height(), data.image.width()
    im_x = (data.width // 2) - (im_width // 2)
    im_y = ((data.height - height) // 2) - (im_height // 2)
    if data.image is not None:
        canvas.create_image(im_x, im_y, anchor=NW, image=data.image)

    # Prediction dialogue
    margin = 30
    canvas.create_text(data.width // 2, data.height - height - margin,
                       anchor=CENTER,
                       font=f"Sans {int(data.width * .05)} bold",
                       fill="red",
                       text=f"{data.predic} at {data.confid*100}% confidence.")


####################################
# Tkinter Animation Template as provided at:
# http://www.krivers.net/15112-s19/notes/notes-animations-part2.html
####################################

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 500   # milliseconds
    root = Tk()
    root.resizable(width=False, height=False)   # prevents resizing window
    init(data)
    # create the root and the canvas
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()
    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed

    count = open("./history/counter.txt", "w")
    count.write(str(data.counter))
    count.close()

    print("bye!")


if __name__ == '__main__':
    main()

