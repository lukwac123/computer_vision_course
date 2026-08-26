import keras
from keras.models import Sequential
from keras.layers import (
    Input,
    Conv2D,
    BatchNormalization,
    MaxPooling2D,
    GlobalAveragePooling2D,
    Dropout,
    Dense
)


class VGGNetSmall:
    def __init__(self, input_shape, num_classes, final_activation="sigmoid"):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation

    def build(self):
        model = Sequential([
            Input(shape=self.input_shape),

            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),

            GlobalAveragePooling2D(),

            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),

            Dense(self.num_classes, activation=self.final_activation)
        ])

        return model


class MobileNetV3Transfer:
    def __init__(self, input_shape, num_classes, final_activation="sigmoid", trainable=False):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.final_activation = final_activation
        self.trainable = trainable

    def build(self):
        base_model = keras.applications.MobileNetV3Small(
            input_shape=self.input_shape,
            include_top=False,
            weights="imagenet"
        )
        base_model.trainable = self.trainable

        inputs = keras.Input(shape=self.input_shape)
        x = base_model(inputs, training=self.trainable)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation="relu")(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(self.num_classes, activation=self.final_activation)(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV3_MultiLabel")
        return model