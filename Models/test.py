import os
import cv2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras_preprocessing.image import img_to_array
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


# Function to extract features using VGG16 model
def extract_features_vgg16(images):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    features = []
    
    for img in images:
        img = cv2.resize(img, (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        feature = model.predict(x)
        features.append(feature.flatten())
    
    return features

test_folder = r'Videos\Test'

test_faces = []

for filename in os.listdir(test_folder):
    if filename.endswith('.mp4'):
        test_faces.extend(extract_faces_from_video(os.path.join(test_folder, filename)))

fake_features = extract_features_vgg16(test_faces)

# Save features to files
os.makedirs('Features', exist_ok=True)

np.save('Features\\test_features.npy', fake_features)

print('Features saved successfully.')



# Load the saved LRCN model
model = load_model('lrcn.h5')

# Load the unknown features from .npy file
unknown_features = np.load('Features\\test_features.npy')

# Reshape and preprocess the unknown features
unknown_features = np.expand_dims(unknown_features, axis=2)

# Make predictions for unknown features
predictions = model.predict(unknown_features)

# Set a threshold for classification
threshold = 0.5

print(predictions)
predictions = np.array(predictions)
avg = np.mean(predictions)

print(avg)