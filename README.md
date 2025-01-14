TITLE: Face Recognition System

Step 1: Setting Up Google Teachable Machine

Go to the Google Teachable Machine website.

<img width="508" alt="image" src="https://github.com/user-attachments/assets/2b5a7598-a959-4c5a-bb2f-496365144b20" />

Select the Image Project option

<img width="840" alt="image" src="https://github.com/user-attachments/assets/25cf4d96-8cc4-46cf-803b-139553f4a758" />

Choose the Standard Image Model or Face Recognition (depending on your goal).

Create classes by capturing images of faces for each class (e.g., different individuals).

<img width="697" alt="image" src="https://github.com/user-attachments/assets/7a8877e4-89fb-483b-8d69-64b51d4b2b5d" />

<img width="286" alt="image" src="https://github.com/user-attachments/assets/fd0696e3-9941-4cb2-a56b-8ff340ca2208" />

<img width="728" alt="image" src="https://github.com/user-attachments/assets/ef8cdd9f-5084-4287-8729-78f3efe12b29" />

Step 2: Exporting the Trained Model

After training, click on the Export Model button.

<img width="570" alt="image" src="https://github.com/user-attachments/assets/977ad975-a4ac-48fc-8ffb-17a5611c181f" />

Select the TensorFlow option.

Download the Keras .h5 model file by clicking on the appropriate export link.

<img width="539" alt="image" src="https://github.com/user-attachments/assets/66402460-5fa8-4ce1-b8b8-712c26a54e7a" />

Step 3: Setting Up the Python Environment

Install Python (if not already installed) by downloading it from python.org. Ensure Python 3.7+ is installed.

Install a code editor like VS Code or PyCharm for coding.

Open a terminal and create a virtual environment:


python -m venv face_rec_env

Activate the virtual environment:
On Windows:

face_rec_env\Scripts\activate
On Mac/Linux:

source face_rec_env/bin/activate

Step 4: Installing Required Libraries

Run the following commands in the terminal to install the necessary libraries:

pip install tensorflow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pillow

Step 5: Loading the Exported Model

Place the .h5 model file downloaded from Teachable Machine in your project directory.
Create a Python script (e.g., face_recognition.py) and add the following code to load the model:
python

import tensorflow as tf

# Load the Teachable Machine Keras model
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully!")

Step 6: Capturing and Preprocessing Images

Use OpenCV to capture images from your webcam:
python

import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Preprocess the images to match the input shape of the Teachable Machine model. Typically, resize them to the required size (e.g., 224x224 pixels).
Description of step6:
This code snippet uses the OpenCV library to capture video from a webcam and display it in real-time. Below is a detailed explanation of each part of the code:

Code Breakdown and Description
1. Import OpenCV Library

import cv2
The cv2 module is imported to use OpenCV, a library for computer vision tasks such as image processing, video capture, and manipulation.

2. Open the Webcam

cap = cv2.VideoCapture(0)
cv2.VideoCapture(0): Opens the default webcam (device index 0). If you have multiple cameras, you can change the index (e.g., 1 for a second camera).
The cap object is used to interact with the webcam for capturing video frames.

3. Infinite Loop to Capture Frames

while True:
    ret, frame = cap.read()
cap.read(): Captures the current frame from the webcam.
ret: A boolean value indicating whether the frame was successfully captured (True or False).
frame: The actual frame captured, represented as a NumPy array.
The loop ensures continuous video capture until stopped.

4. Display the Webcam Feed

cv2.imshow('Webcam', frame)
cv2.imshow(): Displays the captured frame in a window titled "Webcam".
The frame is shown in real-time as a video stream.

5. Exit Condition

if cv2.waitKey(1) & 0xFF == ord('q'):
    break
cv2.waitKey(1): Waits for 1 millisecond for a key press.
If the user presses the 'q' key, the condition becomes True, and the loop exits.
The ord('q') converts the character 'q' into its ASCII code for comparison.

6. Release Resources

cap.release()
cv2.destroyAllWindows()
cap.release(): Releases the webcam resource so it can be used by other applications.
cv2.destroyAllWindows(): Closes all OpenCV windows that were created during the execution of the program.
What the Code Does
Opens the default webcam and starts capturing video frames.
Displays the video stream in real-time in a window named "Webcam".
Allows the user to press the 'q' key to stop the video and close the window.
Potential Applications
Testing Webcam Functionality: Check if the webcam is working.
Base Code for Vision Projects: This is a foundational script for real-time video processing, such as object detection or face recognition.

Step 7: Making Predictions

Add the following code to preprocess the image and make predictions:

python code for face recognition:
from keras.models import load_model
import cv2
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("keras_Model.h5", compile=False)

# Load the labels
class_names = open("labels.txt", "r").readlines()

# Try different backends (CAP_MSMF or CAP_DSHOW) if the default fails
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

while True:
    # Grab the webcam's image
    ret, image = camera.read()

    if not ret:
        print("Error: Failed to capture image from the camera.")
        break

    # Resize the raw image into (224-height,224-width) pixels
    try:
        image_resized = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    except cv2.error as e:
        print(f"OpenCV Error: {e}")
        break

    # Show the image in a window
    cv2.imshow("Webcam Image", image)

    # Make the image a numpy array and reshape it to the model's input shape
    image_array = np.asarray(image_resized, dtype=np.float32).reshape(1, 224, 224, 3)

    # Normalize the image array
    image_array = (image_array / 127.5) - 1

    # Predict using the model
    prediction = model.predict(image_array)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()
    confidence_score = prediction[0][index]

    # Print prediction and confidence score
    print(f"Class: {class_name}, Confidence Score: {np.round(confidence_score * 100, 2)}%")

    # Listen to the keyboard for presses
    keyboard_input = cv2.waitKey(1)

    # 27 is the ASCII for the ESC key on your keyboard
    if keyboard_input == 27:
        print("Exiting program.")
        break

camera.release()
cv2.destroyAllWindows()

 DESCRIPTION OF ABOVE CODE:
 
 Imports Libraries: Loads necessary libraries (keras, cv2, numpy) for machine learning and computer vision tasks.

Load Model and Labels:

Loads a pre-trained Keras model from keras_Model.h5.
Reads class labels from a text file (labels.txt).
Initialize Webcam: Opens the default webcam using OpenCV, with error handling if the webcam fails to initialize.

Real-Time Image Capture:

Continuously captures video frames from the webcam.
Resizes each frame to (224, 224) pixels to match the model’s input size.
Normalizes pixel values to the range [-1, 1].
Model Prediction:

Processes the resized frame through the model to generate predictions.
Finds the class with the highest confidence (argmax).
Displays the predicted class and confidence score in the console.
Display Webcam Feed: Shows the live webcam feed in a window.

Exit Mechanism:

Pressing the ESC key (ASCII 27) stops the loop and closes the webcam feed.
Clean Up:

Releases the webcam and closes all OpenCV windows when exiting.

Creating a face recognition system using Google Teachable Machine and exporting it to a Keras model involves the following steps. Follow these instructions step by step:

Step 1: Setting Up Google Teachable Machine
Go to the Google Teachable Machine website.
Select the Image Project option.
Choose the Standard Image Model or Face Recognition (depending on your goal).
Create classes by capturing images of faces for each class (e.g., different individuals).
Train the model by clicking the Train Model button.

Step 2: Exporting the Trained Model

After training, click on the Export Model button.
Select the TensorFlow option.
Download the Keras .h5 model file by clicking on the appropriate export link.

Step 3: Setting Up the Python Environment

Install Python (if not already installed) by downloading it from python.org. Ensure Python 3.7+ is installed.
Install a code editor like VS Code or PyCharm for coding.
Open a terminal and create a virtual environment:

python -m venv face_rec_env
Activate the virtual environment:
On Windows:

face_rec_env\Scripts\activate
On Mac/Linux:

source face_rec_env/bin/activate

Step 4: Installing Required Libraries

Run the following commands in the terminal to install the necessary libraries:
pip install tensorflow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install pillow

Step 5: Loading the Exported Model

Place the .h5 model file downloaded from Teachable Machine in your project directory.
Create a Python script (e.g., face_recognition.py) and add the following code to load the model:

import tensorflow as tf

# Load the Teachable Machine Keras model
model = tf.keras.models.load_model('model.h5')
print("Model loaded successfully!")

Step 6: Capturing and Preprocessing Images

Use OpenCV to capture images from your webcam:

import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    cv2.imshow('Webcam', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Preprocess the images to match the input shape of the Teachable Machine model. Typically, resize them to the required size (e.g., 224x224 pixels).

Step 7: Making Predictions

Add the following code to preprocess the image and make predictions:


import numpy as np
from PIL import Image

def preprocess_image(image):
    # Resize the image to match model input
    img = Image.fromarray(image)
    img = img.resize((224, 224))  # Replace with the size your model expects
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Make prediction
    predictions = model.predict(processed_frame)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]

    # Display prediction
    cv2.putText(frame, f"Class: {class_idx}, Confidence: {confidence:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
Step 8: Testing and Finalizing:
run the code 

output:

<img width="810" alt="image" src="https://github.com/user-attachments/assets/711d894d-2ca1-4c8f-b6da-d2a0ace059e8" />







TASK2:


TITLE: OFFICE ATTENDENCE SYSTEM

Step 1: Set Up Google Teachable Machine

1.Open Teachable Machine:

<img width="508" alt="image" src="https://github.com/user-attachments/assets/2b5a7598-a959-4c5a-bb2f-496365144b20" />

Visit Teachable Machine.
Select Image Project and then choose the Face Recognition model.

Select the Image Project option

<img width="840" alt="image" src="https://github.com/user-attachments/assets/25cf4d96-8cc4-46cf-803b-139553f4a758" />


Train Your Model:

Create a class for each employee (e.g., "John", "Jane").

<img width="512" alt="image" src="https://github.com/user-attachments/assets/56f66570-680f-4378-bdf6-94bd817a2add" />

Capture multiple images for each employee (10–20 images for accuracy).

<img width="458" alt="image" src="https://github.com/user-attachments/assets/6702a0db-219f-4e3f-9111-43db6b663eda" />

Export the Model:

Click Export Model after training.

<img width="577" alt="image" src="https://github.com/user-attachments/assets/8e88f9c0-d04f-4e10-9d1f-fea0bcd91a09" />

Select TensorFlow.
Download the Keras .h5 model file.


Step 2: Install Required Software

Install Python:

Download and install Python 3.7+ from python.org.

Install a Code Editor:

Use VS Code or PyCharm for coding.

Set Up a Virtual Environment: Open a terminal or command prompt and run

python -m venv attendance_env


Step 3: Install Necessary Libraries


Run these commands in your terminal:
pip install tensorflow
pip install numpy
pip install opencv-python
pip install pillow
pip install mysql-connector-python


Step 4: Set Up the Database (DBMS)

Install MySQL Server:
Download and install MySQL from mysql.com.

Create a Database: Open MySQL Workbench or the terminal and run
CREATE DATABASE office_attendance;
USE office_attendance;

CREATE TABLE attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    employee_name VARCHAR(50),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

Step 5: Write the Python Code
1. Load the Model
Create a script attendance_system.py:

import tensorflow as tf
import cv2
import numpy as np
from mysql.connector import connect

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the class labels
with open("labels.txt", "r") as file:
    class_labels = file.read().splitlines()

2. Connect to the Database
   Add this to the script:
   # Database connection
db = connect(
    host="localhost",
    user="root",  
    password="naveen@123",  
    database="office_attendance"
)
cursor = db.cursor()

3. Capture and Process Webcam Input

   # Open the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break

    # Preprocess the image
    resized_frame = cv2.resize(frame, (224, 224))
    normalized_frame = (np.array(resized_frame) / 127.5) - 1
    input_frame = np.expand_dims(normalized_frame, axis=0)

    # Predict the employee
    predictions = model.predict(input_frame)
    predicted_index = np.argmax(predictions)
    employee_name = class_labels[predicted_index]
    confidence = predictions[0][predicted_index]

    # Display prediction on the screen
    cv2.putText(frame, f"{employee_name} ({confidence:.2f})", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Attendance System', frame)

    # Record attendance if confidence > 80%
    if confidence > 0.8:
        cursor.execute("INSERT INTO attendance (employee_name) VALUES (%s)", (employee_name,))
        db.commit()
        print(f"Attendance marked for {employee_name}")

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
db.close()


Step 6: Run the System
Ensure the .h5 model file and labels.txt are in the same directory as the Python script.
Run the script:
<img width="866" alt="image" src="https://github.com/user-attachments/assets/ae02744d-ad50-4d69-888c-a11f4b157610" />


finally output:
<img width="800" alt="image" src="https://github.com/user-attachments/assets/b8ddd6a8-368a-4329-b80a-8cbb6d44539b" />

Conclusion


Creating an office attendance system using Google Teachable Machine and a DBMS connector combines machine learning and database management to deliver a functional and automated solution. This system enables real-time face recognition to mark attendance, ensuring efficiency and reducing manual errors.

By following the outlined steps:

Model Creation: The Teachable Machine makes it simple to train a robust face recognition model with minimal programming expertise.
Integration: Python and TensorFlow allow seamless integration of the model into a practical application.
Database Logging: Using MySQL ensures reliable storage and easy retrieval of attendance data.
Automation: The webcam feed captures images, processes them in real-time, and logs recognized employees into the database, streamlining the attendance process.
This system is scalable and can be enhanced with features like:

Attendance Reports: Generate daily, weekly, or monthly reports.
Multi-Device Compatibility: Expand to multiple workstations.
Security: Add encryption to protect employee data.
By implementing this project, you gain practical experience in combining machine learning, computer vision, and database systems to solve real-world problems efficiently.

























