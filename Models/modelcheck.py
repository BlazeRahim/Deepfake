from tensorflow.keras.models import load_model
model = load_model('lrcn.h5')
print(model.summary())