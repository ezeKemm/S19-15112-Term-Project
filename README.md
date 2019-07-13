# S19-15112-Term-Project
# Recycling Robot
This program uses machine learning to create a system which identifies types of trash 
and sorts the trash into the appropiate bins based on the material type.
This allows human users to avoid the inconvenience of determining whether a material can be recycled 
or not while allowing detailed and complex recycling to still occur. 
A GUI allows users to see the prediction and correct mistakes before sorting as this model is a prototype. 

Currently, this prototype distinguishes 5 material classes: Plastic, Cardboard, Plastic, Glass, Metal; 
with a final Trash category for all materials not categorized as any of the 5 classes. 
This material identification is completed using transfer learning to distinguish visual characteristics of 
the material characteristics for image classification. 

# HOW TO

-- TRAINING
To begin, a model must be trained on the provided image dataset before making predictions. 
Loading ML_TrainingProgram.py will train a pre-trained model from dataset-resized.zip
Once trained, the model will be exported to be used in the primary program. 
If a new model is trained, be sure to remove the previously saved model so the program does not pick the wrong model

-- HARDWARE
This program interfaces with an Arduino microcontroller to control the sorting mechanism. 
An Arduino board (preferrably an UNO), is required as well as two servo motors.
The Arduino IDE is required to load the Arduino program serial_servo_test1.ino to the board.
Once loaded, this program is no longer needed unless a new program is loaded on the board.
The pins used to control the motors are pins 9 and 10 and can be changed in the setup() of serial_servo_test1.ino

The serial port the Arduino is connected to is also required for the program and can be found in the Arduino IDE
under Tools -> Port. The needed port will be listed next to the connected board name, take this port name and
place it in the port variable in UI_Interface.py under main()

A camera is also required for this program, by default the program accesses VideoCapture(0), 
which is usually the default webcam of the connected computer.

-- RUNNING
Once all of the above is confirmed, the project can be started from UI_Interface.py 
Upon successful connection with hardware and loading of trained model, a GUI will appear showing all material categories.
Pressing 'e' will take a snapshot using the camera, displaying the image and the prediction. 
Once displayed, users may verify if the prediction is correct or not by selecting which material class the item actually is,
the machine will log this information and then sort the item into the appropiate bin.  
Pressing 'e' again will repeat this process until the program is terminated. 

**If users wish to test the program functionality without the Arduino hardware, 
simply remove the serial.Serial() command under main() and the sort() command under check_cursor().
This prevents establishing a serial connection and attempting to control the servos after verification. 


# Required Modules
- fastAI
- pandas
- numpy
- matplotlib
- sklearn
- os
- pathlib
- zipfile
- shutil
- re
- seaborn
- tkinter
- PIL
- pyserial
- OpenCV

