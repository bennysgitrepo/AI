# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import face_recognition
from keras.models import load_model
from keras.preprocessing import image
from scipy.io import loadmat
from statistics import mode
from random import shuffle

# Define detect_faces function and two parameters:
# Return the value from detection model in grayscale (value of 1,3 and 5)
def detect_faces(detection_model, gray_image_array):
    return detection_model.detectMultiScale(gray_image_array, 1.3, 5)

# Define draw_text function and eight parameters:
# 	Set the coordinates for the box to appear 
# 	Use built-in command from cv2 library to set the font scale and colour
def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_PLAIN,
                font_scale, color, thickness, cv2.LINE_AA)

# Define draw_bounding_box function and three parameters:
# 	Set coordinates for face detection
# 	Use built-in command from cv2 library to draw a rectangle once face is detected
def draw_bounding_box(face_coordinates, image_array, color):
    x, y, w, h = face_coordinates
    cv2.rectangle(image_array, (x, y), (x + w, y + h), color, 2)

# Define apply_offset function and two parameters:
# 	Set coordinates for face detection
# 	Set values for offset
# 	Return values from the function
def apply_offsets(face_coordinates, offsets):
    x, y, width, height = face_coordinates
    x_off, y_off = offsets
    return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

# Define load_detection_model function and one parameter:
#   Use built-in command from cv2 library to load the Cascade classifier
# 	Return values from the function
def load_detection_model(model_path):
    detection_model = cv2.CascadeClassifier(model_path)
    return detection_model

# Define get_labels and one parameter:
# 	If datasetname is ‘fer2013’:
#       Return values of the emotions in array of numbers (e.g. 0 = ‘Angry’) from the function
# 	Else
# 		Raise an error exception and print error message
def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
                4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    else:
        raise Exception('dataset name incorrect, or not specified')

# Define preprocess_input and two parameters:
# 	Define a float variable and allow a numeric value to pass 
# 	If v2 remains True:
# 		Pass the numeric value and subtract it by 0.5
# 		Pass the corresponding value and multiple by 2.0
# 	Return the numeric value from the function 
def preprocess_input(x, v2=True):
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

# Define detection_model variable and load the Haar Cascade model dataset
# Define emotion_model variable and load the emotion model dataset
# Call get_labels function to load fer2013 dataset
detection_model = 'haarcascade_frontalface_default.xml'
emotion_model = 'emotion_model.hdf5'
emotion_labels = get_labels('fer2013')

# Set frame window and emotion offsets values
frame_window = 10
emotion_offsets = (20, 40)

# Call load_detection_model function
# Load emotion model
face_detection = load_detection_model(detection_model)
emotion_classifier = load_model(emotion_model, compile=False)

# Enable the model and prepare the dataset for the inference
emotion_target_size = emotion_classifier.input_shape[1:3]

# Create an empty list to calculate modes in the model
emotion_window = []

# Create the ‘Facial and Emotion Detection’ window
# Enable video capture using cv2 library
cv2.namedWindow('Facial and Emotion Detection')
video_capture = cv2.VideoCapture(0)

# Import Ben and Sashinka’s image for the face recognition model
# Set the face encodings for the imported images
ben_image = face_recognition.load_image_file('Ben.jpg')
sashinka_image = face_recognition.load_image_file('Sashinka.jpg')
ben_face_encoding = face_recognition.face_encodings(ben_image)[0]
sashinka_face_encoding = face_recognition.face_encodings(sashinka_image)[0]

# Create an array of Ben and Sashinka’s face encodings for identification
# Create an array of ID names when known faces are recognised
id_face_encodings = [ben_face_encoding, sashinka_face_encoding]
id_face_names = ['Ben Walker', 'Sashinka Lintang']

# Define open variables for face detection 
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

# While Loop (terminating program when user presses ‘q’):
while True:

	# Capture a frame from video input
	# Define variable to convert image into greyscale
	# Define variable to convert image into RGB
	# Call the detect_faces function to process image in greyscale
    bgr_image = video_capture.read()[1]
    gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    faces = detect_faces(face_detection, gray_image)

    # Resize frame to optimise computational performance
    small_frame = cv2.resize(bgr_image, (0, 0), fx=0.25, fy=0.25)

    # Convert greyscale image into RGB
	# Resize RGB frame to optimise computational performance 
    rgb_small_frame = small_frame[:, :, ::-1]

    # If process_this_frame is True:
    if process_this_frame:

        # Store face and the encodings in the single frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # Initialise an empty face names list for the output
        face_names = []

        # For each face encodings in the face_encodings array:
        for face_encoding in face_encodings:

		    # Compare faces in the encodings with the known encodings		
            # and print “Unknown Individual” if there’s no match
            matches = face_recognition.compare_faces(id_face_encodings, face_encoding)
            name = "Unknown Individual"

            # Define best match index by looking for the minimum element in the array
            # If there’s a match between best match index and match, then append name into face names list
            face_distances = face_recognition.face_distance(id_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = id_face_names[best_match_index]

            face_names.append(name)

    # Stops the processing of the frame by setting it to False
    process_this_frame = not process_this_frame

    # For each face identified in the frame:
	#   Draw a rectangle box
	#   Display output of name(s)
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.rectangle(rgb_image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(rgb_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # For each recognised emotion in the frame:
    # 	Try to convert frame into grayscale
	# Except:
	# 	Continue if failed to convert frame
    for face_coordinates in faces:
        x1, x2, y1, y2 = apply_offsets(face_coordinates, emotion_offsets)
        gray_face = gray_image[y1:y2, x1:x2]
        try:
            gray_face = cv2.resize(gray_face, (emotion_target_size))
        except:
            continue

        # Compare the frame into loaded emotion dataset to define 
        # detected emotion, and append to the emotion window list
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        
        emotion_prediction = emotion_classifier.predict(gray_face)
        emotion_probability = np.max(emotion_prediction)
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text = emotion_labels[emotion_label_arg]
        emotion_window.append(emotion_text)

	    # If the emotion window > frame window:
		#     Return name of emotion into the emotion window list
	    # Try to store the value in the emotion window list for output
		#     Except:
		# 	    Continue if failed to store value in the list
        if len(emotion_window) > frame_window:
            emotion_window.pop(0)
        try:
            emotion_mode = mode(emotion_window)
        except:
            continue

        # If emotion is ‘angry’, ‘disgust’, fear, or ‘sad’:
        #     Display red coloured window and print “Threat Rating: Threatening”
	    # Else if emotion is ‘happy’, ‘surprise’, or ‘neutral’:
        #     Display green coloured window and print “Threat Rating: Non-Threatening”
	    # Else:
        #     Display blue coloured window and print “Threat Rating: Unknown”
        if emotion_text in ('Angry', 'Disgust', 'Fear', 'Sad'):
            color = emotion_probability * np.asarray((255, 0, 0))
            threat_rating = ("Threat Rating: Threatening")
        elif emotion_text in ('Happy', 'Surprise', 'Neutral'):
            color = emotion_probability * np.asarray((0, 255, 0))
            threat_rating = ("Threat Rating: Non-Threatening")
        else:
            color = emotion_probability * np.asarray((0, 0, 255))
            threatRating = ("Threat Rating: Unknown")

        color = color.astype(int)
        color = color.tolist()

        draw_bounding_box(face_coordinates, rgb_image, color)
        draw_text(face_coordinates, rgb_image, emotion_mode, color, 0, -45, 1, 1)
        draw_text(face_coordinates, rgb_image, threat_rating, color, 0, -70, 1, 1)

    # Display output to user’s screen
    bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.imshow('Facial and Emotion Detection', bgr_image)

    # If user presses ‘q’ at any point:
	#   Break from loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Disable video capture and destroy all windows
video_capture.release()
cv2.destroyAllWindows()