import tensorflow as tf
from tensorflow.keras import layers, models

def build_segnet(input_shape=(256, 256, 3), num_classes=21):
    inputs = layers.Input(shape=input_shape)
    
    # Encoder (VGG16-style)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x, p1 = layers.MaxPooling2D((2, 2), strides=(2, 2), return_indices=True)(x)
    
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x, p2 = layers.MaxPooling2D((2, 2), strides=(2, 2), return_indices=True)(x)

    # Decoder
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    
    # Pixel-wise Classification
    outputs = layers.Conv2D(num_classes, (1, 1), activation="softmax")(x)
    
    model = models.Model(inputs, outputs)
    return model

# Initialize model
model = build_segnet()
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()

