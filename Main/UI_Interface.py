from tkinter import *
from PIL import Image, ImageTk
import time
import cv2 as cv
from Main import sortingAlgorithm_TechDemo as capture


# TODO -- Link file to be used by sortingAlgorithm
####################################
# customize these functions
####################################


# TODO -- Rework to not use global vars
def start_UI(classes):
    global num_classes
    global labels
    labels = classes
    num_classes = len(classes)
    run(600, 300)


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
def check_Cursor(x, y, data):

    height = data.height * data.bcell_height

    for button in range(data.num_buttons):
        start_x = button * data.bcell_width + data.x_margin
        end_x = button * data.bcell_width + data.bcell_width - data.x_margin
        start_y = data.height - height + data.y_margin
        end_y = data.height - data.y_margin

        if start_x <= x <= end_x and start_y <= y <= end_y:
            print(f"Click detected at Button {button+1}")
            data.button_fill[button] = "red"


def retrieve_image():

    # Test code
    # TODO -- Replace with call to webcam capture
    path = ""
    # image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    image = Image.open(path)
    image = ImageTk.PhotoImage(image)
    # image = PhotoImage(file=path)
    # image = Image.fromarray(image)
    print("Success!")

    return image


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
    pass


def mousePressed(event, data):
    # use event.x and event.y
    check_Cursor(event.x, event.y, data)
    pass


def keyPressed(event, data):
    # use event.char and event.keysym
    if event.keysym == "e":
        retrieve_image()
    pass


def timerFired(data):
    for button in range(data.num_buttons):
        if data.button_fill[button] == "red":
            data.button_fill[button] = "white"


def redrawAll(canvas, data):
    # draw in canvas

    # TODO -- Center buttons correctly
    # Draw panel background
    # height = data.height * data.bcell_height
    # canvas.create_rectangle(0, 0, data.width, data.height - height, fill="gray")
    # canvas.create_rectangle(0, data.height - height, data.width, data.height, fill="black")

    # # Draw Buttons
    # for button in range(data.num_buttons):
    #     start_x = button * data.bcell_width + data.x_margin
    #     end_x = button * data.bcell_width + data.bcell_width - data.x_margin
    #     start_y = data.height - height + data.y_margin
    #     end_y = data.height - data.y_margin
    #
    #     # TODO -- Remove this and fix calculation - Clicking out of bounds still works
    #     if button == 0:
    #         start_x = button * data.bcell_width + (data.x_margin * 2)
    #     elif button == 3:
    #         end_x = button * data.bcell_width + data.bcell_width - (data.x_margin * 2)
    #
    #     canvas.create_rectangle(start_x, start_y, end_x, end_y,
    #                             fill=data.button_fill[button])
    #
    #     # Adds text labels to buttons
    #     txt_x = (end_x - start_x) // 2 + start_x
    #     txt_y = (height // 2) + (data.height - height) - data.y_margin    # Sloppy
    #     # TODO -- Try to find simpler calculation
    #     canvas.create_text(txt_x, txt_y,
    #                        text=data.button_labels[button],
    #                        anchor=N,
    #                        font="Sans 10 bold")

    image = retrieve_image()

    canvas.create_image(20, 20, anchor=NW, image=image)

        ## DEBUG

        # start_x2 = button * data.bcell_width
        # end_x2 = button * data.bcell_width + data.bcell_width
        # start_y2 = data.height - height
        # end_y2 = data.height
        #
        # colors = ["green", "orange", "pink", "yellow"]
        # canvas.create_rectangle(start_x2, start_y2, end_x2, end_y2, fill="", outline=colors[button])


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


start_UI(["Paper/Cardboard", "Plastic", "Glass/Metal", "Trash"])


