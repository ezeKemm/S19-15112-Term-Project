from __future__ import print_function, division

from tkinter import *
import cv2 as cv
from PIL import Image, ImageTk
import time
import cv2 as cv
# from Main import sortingAlgorithm_TechDemo as capture


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
    # path = os.getcwd() + "/data"

    path = "/Users/zeke/PycharmProjects/15112_TP_S19/SortingTraining/data"
    tfms = get_transforms(do_flip=False)
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
        except:
            print("Failed to connect to webcam, retrying...")
            continue
        else:
            break
    else:
        print("Failed to successfully connect to camera after 10 tries")
        quit()

    return webcam

####################################
####################################


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


# Captures webcam output, converts to be displayed on UI
def take_shot():
    __, image = webcam.read()
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.fromarray(image)
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


# Determines if click is in the buttons' boundaries
def check_cursor(x, y, data):

    height = data.height * data.bcell_height

    for button in range(data.num_buttons):
        start_x = button * data.bcell_width + data.x_margin
        end_x = button * data.bcell_width + data.bcell_width - data.x_margin
        start_y = data.height - height + data.y_margin
        end_y = data.height - data.y_margin

        if start_x <= x <= end_x and start_y <= y <= end_y:
            print(f"Click detected at Button {button+1}")
            data.button_fill[button] = "red"


def main():

    # Loads image classifying model and retrieves webcam capture
    global learn, webcam, num_classes, labels
    classes = ["Paper/Cardboard", "Plastic", "Glass/Metal", "Trash"]

    learn = load_model()
    webcam = connect_capture_device()
    labels = classes
    num_classes = len(classes)
    run(600, 300)


def init(data):
    # load data.xyz as appropriate
    data.timer = 0
    data.num_buttons = num_classes
    data.bcell_width = data.width // data.num_buttons
    data.bcell_height = 0.30    # represented as a fraction of total height
    data.x_margin = 5
    data.y_margin = 10
    data.button_fill = [ "white" for i in range(data.num_buttons) ]
    data.button_labels = labels

    data.image = None
    pass


def mousePressed(event, data):
    # use event.x and event.y
    check_cursor(event.x, event.y, data)
    pass


def keyPressed(event, data):
    # use event.char and event.keysym
    if event.keysym == "e":
        data.image = take_shot()
    pass


def timerFired(data):
    for button in range(data.num_buttons):
        if data.button_fill[button] == "red":
            data.button_fill[button] = "white"


def redrawAll(canvas, data):

    # TODO -- Center buttons correctly
    # Draw panel background
    height = data.height * data.bcell_height
    canvas.create_rectangle(0, 0, data.width, data.height - height, fill="gray")
    canvas.create_rectangle(0, data.height - height, data.width, data.height, fill="black")

    # Draw Buttons
    for button in range(data.num_buttons):
        start_x = button * data.bcell_width + data.x_margin
        end_x = button * data.bcell_width + data.bcell_width - data.x_margin
        start_y = data.height - height + data.y_margin
        end_y = data.height - data.y_margin

        # TODO -- Remove this and fix calculation - Clicking out of bounds still works
        if button == 0:
            start_x = button * data.bcell_width + (data.x_margin * 2)
        elif button == 3:
            end_x = button * data.bcell_width + data.bcell_width - (data.x_margin * 2)

        canvas.create_rectangle(start_x, start_y, end_x, end_y,
                                fill=data.button_fill[button])

        # Adds text labels to buttons
        txt_x = (end_x - start_x) // 2 + start_x
        txt_y = (height // 2) + (data.height - height) - data.y_margin    # Sloppy
        # TODO -- Try to find simpler calculation
        canvas.create_text(txt_x, txt_y,
                           text=data.button_labels[button],
                           anchor=N,
                           font="Sans 10 bold")

    if data.image is not None:
        canvas.create_image(20, 20, anchor=NW, image=data.image)


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
    print("bye!")


if __name__ == '__main__':
    main()


