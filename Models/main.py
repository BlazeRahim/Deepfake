import os
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import tensorflow as tf
import cv2
from keras_preprocessing.image import img_to_array

a = 0


def extract_faces_from_video(video_path):
    global a
    print(f"Extracting Vid No. {a}")
    a += 1
    cap = cv2.VideoCapture(video_path)

    # Initialize the CPU-based face detector
    face_cascade_cpu = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    faces = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % fps == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces on CPU
            detected_faces = face_cascade_cpu.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in detected_faces:
                face = frame[y:y + h, x:x + w]
                face = cv2.resize(face, (224, 224))
                faces.append(face)

        frame_count += 1

    cap.release()
    return faces


def extract_features_vgg16(images):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')

    features = []

    for img in images:
        img = cv2.resize(img, (224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        with tf.device('/GPU:0'):
            feature = model.predict(x)

        features.append(feature.flatten())

    return features


real_folder = r'Videos\Real'
fake_folder = r'Videos\Fake'

real_faces = []
fake_faces = []

for filename in os.listdir(real_folder):
    if filename.endswith('.mp4'):
        real_faces.extend(extract_faces_from_video(os.path.join(real_folder, filename)))

for filename in os.listdir(fake_folder):
    if filename.endswith('.mp4'):
        fake_faces.extend(extract_faces_from_video(os.path.join(fake_folder, filename)))

real_features = extract_features_vgg16(real_faces)
fake_features = extract_features_vgg16(fake_faces)

os.makedirs('Features', exist_ok=True)

np.save('Features\\real_features.npy', real_features)
np.save('Features\\fake_features.npy', fake_features)

print('Features saved successfully.')
