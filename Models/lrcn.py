import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Conv1D, MaxPooling1D, Flatten
from sklearn.model_selection import train_test_split

# Load the saved .npy files
real_features = np.load('Features\\real_features.npy')
fake_features = np.load('Features\\fake_features.npy')

# Create labels
real_labels = np.ones(len(real_features))
fake_labels = np.zeros(len(fake_features))

# Concatenate features and labels
X = np.concatenate((real_features, fake_features), axis=0)
y = np.concatenate((real_labels, fake_labels), axis=0)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape features for Conv1D input
X_train = np.expand_dims(X_train, axis=2)
X_test = np.expand_dims(X_test, axis=2)

# Define LRCN model
model = Sequential()

# Conv1D layer
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(512, 1)))
model.add(MaxPooling1D(pool_size=2))

# LSTM layer
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.5))

# Flatten layer
model.add(Flatten())

# Dense layers
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=16)

model.save('lrcn.h5')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
