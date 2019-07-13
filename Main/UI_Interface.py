from __future__ import print_function, division

from tkinter import *
from tkinter import messagebox
from PIL import Image, ImageTk

import serial
import cv2 as cv
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
    # To retrieve absolute path, moves to parent directory and retrieves correct path
    os.chdir('..')
    abspath = os.path.abspath("./SortingTraining/data")
    data = ImageDataBunch.from_folder(abspath, train="train", valid="valid")

    # Loads trained model, exports for inference, and loads inference model
    learn = cnn_learner(data, models.resnet34).load('recyc_model')
    learn.export()
    learn = load_learner(abspath)

    os.chdir('./Main')  # Resets cwd to Main
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
            return webcam
        except:
            print("Failed to connect to webcam, retrying...")
            continue
    else:
        print("Failed to successfully connect to camera after 10 tries")
        quit()


# Creates directory to store data history on results
def load_folder():
    path = "./history"

    try:
        if not os.path.exists(path):
            os.makedirs(path)
            return path
    except OSError:
        print("Couldn't create directory")


# Starts program, loading model and capture device
def main():
    global ser, learn, webcam, classes, hist_path

    classes = ["Paper", "Cardboard", "Plastic", "Glass", "Metal", "Trash"]
    port = '/dev/cu.usbmodem14301'   # Must determine port Arduino is connected (MacOS port)
    ser = serial.Serial(port, 9600)
    learn = load_model()
    webcam = connect_capture_device()
    hist_path = load_folder()

    run(500, 400)


####################################
####################################

# Classifies image using machine learning,
# determines where item should be disposed of
def prediction(input, data):

    # Forward propagates through nn to classify image
    pred, idx, probs = learn.predict(Image(pil2tensor(input, np.float32).div_(255)))
    print(str(pred).capitalize(), float(probs[idx]), probs)

    # If accurate prediction can't be made, chooses trash
    if max(probs) < 0.5:
        data.pred = "Trash"
        data.conf = "0"
    else:
        data.pred = str(pred).capitalize()
        data.conf = round(float(probs[idx]), 2)

    data.buttons[data.pred]["num_predicted"] += 1   # Updates counter


# Captures webcam output, converts to be displayed on GUI
def take_shot(data):
    data.counter += 1
    __, image = webcam.read()
    prediction(image, data)
    # Save image
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


# Updates log used for tracking program results
def update_log(correct, data):
    log = open("./history/prediction_log.txt", "a")
    prediction_result = ""

    if correct == data.pred:
        prediction_result = f"Predicted {data.pred}...Correct: True"
    elif correct != data.pred:
        prediction_result = f"Predicted {data.pred}...Correct: False -> {correct}"

    entry = f"#{data.counter} -- capture{data.counter}.jpeg -- " + prediction_result + "\n"
    print(entry)

    log.write(entry)
    log.close()


# Updates clicked_status
def update_status(data, button):
    d = data.buttons
    for b in data.buttons:
        if b != button:
            d[b]["clicked_status"] = False
        else:
            d[button]["clicked_status"] = True


# Sends signal to arduino to sort item based off material
def sort(item):
    print(item)
    if item == "Paper" or item == "Cardboard":
        print("Sending to paper/cardboard")
        ser.write("0".encode())
        print(str(ser.readline().decode()))
    elif item == "Plastic":
        print("sending to plastic")
        ser.write("1".encode())
        print(str(ser.readline().decode()))
    elif item == "Glass" or item == "Metal":
        print('sending to glass/metal')
        ser.write("2".encode())
        print(str(ser.readline().decode()))
    elif item == "Trash":
        print('sending to trash')
        ser.write("3".encode())
        print(str(ser.readline().decode()))


# Determines if click is in the buttons' boundaries
def check_cursor(x, y, data):
    c, d = "coord", data.buttons    # Aliasing for shorthand

    for button in data.buttons:
        x1, y1, x2, y2 = d[button][c][0], d[button][c][1], d[button][c][2], d[button][c][3]

        if x1 <= x <= x2 and y1 <= y <= y2 and not data.pred_check:

            # Highlights based on prediction verification
            if button == data.pred:    # Prediction correct
                d[button]["button_color"] = "DeepSkyBlue2"
                d[button]["txt_color"] = "white"
                d[data.pred]["num_correct"] += 1
                update_log(button, data)

            elif button != data.pred:    # Prediction incorrect
                d[button]["button_color"] = "red"
                update_log(button, data)

            update_status(data, button)
            data.pred_check = True    # Changes state after verification to allow capture
            sort(button)

        # Error message
        elif x1 <= x <= x2 and y1 <= y <= y2 and data.pred_check:
            messagebox.showerror("Invalid Action",
                                 """Prediction already verified, please create a new capture"""
                                 )


# This generates the dictionary used to store all values for the
def generate_buttons(labels):
    dictn = {}
    default_button_color = "white"
    for button in labels:
        dictn[button] = {
            "name": button,
            "coord": [],
            "txt_coord": (0, 0),
            "button_color": default_button_color,
            "txt_color": "black",
            "clicked_status": False,
            "num_predicted": 0,
            "num_correct": 0
        }

    return dictn


def draw_buttons(data):
    # Draw Buttons
    i = 0
    for button in data.buttons:
        start_x = i * data.bwidth + data.margin
        end_x = i * data.bwidth + data.bwidth - data.margin
        start_y = data.bheight + data.margin * 2
        end_y = data.height - data.margin * 2

        # Ensures margins are even on panel ends
        if i == 0:
            start_x = i * data.bwidth + (data.margin * 2)
        elif i == 3:
            end_x = i * data.bwidth + data.bwidth - (data.margin * 2)

        data.buttons[button]["coord"] = [start_x, start_y, end_x, end_y]

        # Adds text labels to buttons
        txt_x = (end_x - start_x) // 2 + start_x
        txt_y = (end_y - start_y) // 2 + start_y

        data.buttons[button]["txt_coord"] = (txt_x, txt_y)

        i += 1


# Refreshes all buttons after new image capture
def refresh_buttons(data):
    for b in data.buttons:
        data.buttons[b]["button_color"] = "white"
        data.buttons[b]["txt_color"] = "black"

        if b == data.pred:
            data.buttons[b]["button_color"] = "green"


def init(data):

    # Parameters for the buttons
    data.buttons = generate_buttons(classes)
    data.bwidth = data.width // len(data.buttons)
    data.bheight = data.height - data.height * 0.2
    data.margin = 5

    data.counter = 0
    data.image = None
    data.pred = "Glass"
    data.conf = 0

    data.pred_check = True  # State determines valid actions
    draw_buttons(data)  # Determines coordinates of all buttons


def mousePressed(event, data):
    # use event.x and event.y
    check_cursor(event.x, event.y, data)


def keyPressed(event, data):
    # use event.char and event.keysym
    # Captures image from device on press
    if data.pred_check and event.keysym == "e":
        data.pred_check = False
        data.image = take_shot(data)
        refresh_buttons(data)

    elif event.keysym == "e" and not data.pred_check:
        messagebox.showerror("Invalid Action",
                             """Must verify current prediction before taking new capture"""
                             )


def timerFired(data):
    pass


def redrawAll(canvas, data):

    # Draw panel background
    canvas.create_rectangle(0, 0, data.width, data.bheight, fill="black")   # Upper panel
    canvas.create_rectangle(0, data.bheight, data.width, data.height, fill="gray")  # Lower/Button panel

    # Draws buttons with text labels
    for b in data.buttons:
        c, d, t = "coord", data.buttons, "txt_coord"

        canvas.create_rectangle(d[b][c][0], d[b][c][1], d[b][c][2], d[b][c][3],
                                fill=d[b]["button_color"])
        canvas.create_text(d[b][t][0], d[b][t][1],
                           text=b,
                           fill=d[b]["txt_color"],
                           anchor=CENTER,
                           font=f"Sans {int(data.bwidth * .1)} bold")

    # Displays images from capture device
    if data.image is not None:
        im_height, im_width = data.image.height(), data.image.width()
        im_x = (data.width // 2) - (im_width // 2)
        im_y = (data.bheight // 2) - (im_height // 2)
        canvas.create_image(im_x, im_y, anchor=NW, image=data.image)

    # Prediction dialogue
    margin = 30
    canvas.create_text(data.width // 2, data.bheight - (margin * 2),
                       anchor=CENTER,
                       font=f"Sans {int(data.width * .05)} bold",
                       fill="white",
                       text=f"{data.pred} at {data.conf*100}% confidence.")


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
    data.timerDelay = 100   # milliseconds
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

    print("bye!")


if __name__ == '__main__':
    main()

