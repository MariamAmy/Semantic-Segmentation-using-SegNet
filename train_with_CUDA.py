import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from segnet_model import build_segnet

# Enable CUDA
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    print(f"Running on GPU: {physical_devices[0]}")
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
else:
    print("No GPU detected. Running on CPU.")

# Load and preprocess the dataset (VOC2012)
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    'data/VOC2012/train',
    target_size=(256, 256),
    batch_size=32,
    class_mode='sparse'
)

# Build and train the model
model = build_segnet()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(train_generator, epochs=10)
model.save("segnet_voc2012.h5")

