import tensorflow as tf

# Create a TensorFlow session
sess = tf.compat.v1.Session()

# Get the list of GPU devices available to TensorFlow
gpu_devices = sess.list_devices()

# Iterate over the list of devices and print their details
for device in gpu_devices:
    print(device)
