from keras.applications.vgg16 import VGG16, preprocess_input
import cv2
import numpy as np
from keras.preprocessing import image

# Function to extract features using VGG16 model
def extract_features_vgg16(images):
    model = VGG16(weights='imagenet', include_top=False, pooling='avg')
    
    features = []
    
    for img in images:
        img = cv2.resize(img, (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        feature = model.predict(x)
        features.append(feature.flatten())
    
    return features
