# Import libraries into the main program
import cv2
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
import face_recognition
from keras.models import load_model
from keras.preprocessing import image
from scipy.io import loadmat
from statistics import mode
from random import shuffle

# Define functions for the emotion detection in the main program
def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale, color, thickness, cv2.LINE_AA)

def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    else:
        raise Exception('dataset name incorrect, or not specified')

def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# Loading data sources for emotion detection into main program
detection_model_path = 'haarcascade_frontalface_default.xml'
emotion_model_path = 'emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# Defining parameters for the box shapes
frame_window = 10
emotion_offsets = (20, 40)

# Loading data models
face_detection = load_detection_model(detection_model_path)
emotion_classifier = load_model(emotion_model_path, compile=False)

# Enabling input model shapes for inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Enabling list for calculating modes
emotion_window = []

# Enabling the video streaming component and supporting library for facial & emotion detection
cv2.namedWindow('Facial and Emotion Detection')
video_capture = cv2.VideoCapture(0)

# Enable face_detection library, and import images of the two individuals (Ben Walker & Sashinka Lintang) for the main program to learn
ben_image = face_recognition.load_image_file('Ben.jpg')
sashinka_image = face_recognition.load_image_file('Sashinka.jpg')
ben_face_encoding = face_recognition.face_encodings(ben_image)[0]
sashinka_face_encoding = face_recognition.face_encodings(sashinka_image)[0]

# Build the array/list of faces encodings and names 
id_face_encodings = [ben_face_encoding, sashinka_face_encoding]
id_face_names = ['Ben Walker', 'Sashinka Lintang']

# Initialise the open variables for face detection
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# While the main program is executed (if 'q' is pressed, disable the main program), run in parallel the face detection and emotion detection functions
while True:

# Capture a frame of video
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

# Optimise the frame for performance and facial recognition
    small_frame = cv2.resize(bgr_image, (0, 0), fx=0.25, fy=0.25)

# Convert BGR color to RGB
    rgb_small_frame = small_frame[:, :, ::-1]

    if process_this_frame:

        # Find the faces and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            # If the face is match, print the name of the individual; else print "Unknown Individual"
            matches = face_recognition.compare_faces(id_face_encodings, face_encoding)
            name = "Unknown Individual"

            face_distances = face_recognition.face_distance(id_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = id_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame

# Output the results of the facial detection
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Display the rectangle box
        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 0, 255), 2)

        # Display the name of the individual
        cv2.rectangle(rgb_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(rgb_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# The emotion detection program
    for face_coordinates in faces:

        # Convert frame to grey-scale, and import predictive models into the program
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # If the emotion found is either in the list below, display the output in a coloured window
        if emotion_text in ('Angry', 'Disgust', 'Fear', 'Sad'):
            color = emotion_probability * np.asarray((255, 0, 0))
        elif emotion_text in ('Happy', 'Surprise', 'Neutral'):
            color = emotion_probability * np.asarray((0, 255, 0))
        else:
            color = emotion_probability * np.asarray((0, 0, 255))

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)

    # Output the result to screen
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Facial and Emotion Detection', bgr_image)

    # Quit the main program by pressing "q" on the keyboard
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Disable video capture
video_capture.release()
cv2.destroyAllWindows()