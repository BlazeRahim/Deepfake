import numpy as np

# Paths to the saved .npy files
real_features_path = 'Features\\real_features.npy'
fake_features_path = 'Features\\fake_features.npy'

# Load the saved .npy files
real_features = np.load(real_features_path)
fake_features = np.load(fake_features_path)

# Check the dimensions
real_shape = real_features.shape
fake_shape = fake_features.shape

print(f"Dimensions of real_features: {real_shape}")
print(f"Dimensions of fake_features: {fake_shape}")
