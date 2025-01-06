import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from segnet_model import build_segnet

# Load the trained model
model = tf.keras.models.load_model("segnet_voc2012.h5")

# Load a test image
img_path = 'data/VOC2012/test/sample.jpg'
img = image.load_img(img_path, target_size=(256, 256))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction[0], axis=-1)

# Display results
import matplotlib.pyplot as plt
plt.imshow(predicted_class)
plt.title("Segmented Output")
plt.show()

