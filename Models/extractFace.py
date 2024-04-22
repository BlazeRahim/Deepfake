import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model


# Function to extract faces from videos
def extract_faces_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    faces = []
    
    # Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Initialize counter for skipping frames
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Skip frames to capture only 1 image per second
        if frame_count % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
            
            for (x, y, w, h) in detected_faces:
                face = frame[y:y+h, x:x+w]
                face = cv2.resize(face, (224, 224))
                faces.append(face)
        
        frame_count += 1
    
    cap.release()
    return faces
