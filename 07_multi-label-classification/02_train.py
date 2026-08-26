from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from imutils import paths

import numpy as np
import pandas as pd
import argparse
import pickle
import cv2
import os


# ---------------------------------------------------------
# Ograniczenie komunikatów TensorFlow / Keras
# ---------------------------------------------------------

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ---------------------------------------------------------
# Keras & TensorFlow
# ---------------------------------------------------------

import tensorflow as tf
import keras

from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from architecture import models


# ---------------------------------------------------------
# Wykres historii treningu
# ---------------------------------------------------------

def plot_hist(history, filename):

    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Binary Accuracy",
            "F1 Score (Macro)",
            "Loss"
        )
    )

    # Accuracy - trening & walidacja
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["binary_accuracy"],
            name="train_binary_accuracy",
            mode="markers+lines"
        ),
        row=1,
        col=1
    )

    if "val_binary_accuracy" in hist.columns:
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["val_binary_accuracy"],
                name="valid_binary_accuracy",
                mode="markers+lines"
            ),
            row=1,
            col=1
        )

    # F1 Macro - trening & walidacja
    if "f1_macro" in hist.columns:
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["f1_macro"],
                name="train_f1_macro",
                mode="markers+lines"
            ),
            row=2,
            col=1
        )
    if "val_f1_macro" in hist.columns:
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["val_f1_macro"],
                name="valid_f1_macro",
                mode="markers+lines"
            ),
            row=2,
            col=1
        )

    # Loss - trening & walidacja
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["loss"],
            name="train_loss",
            mode="markers+lines"
        ),
        row=3,
        col=1
    )

    if "val_loss" in hist.columns:
        fig.add_trace(
            go.Scatter(
                x=hist["epoch"],
                y=hist["val_loss"],
                name="valid_loss",
                mode="markers+lines"
            ),
            row=3,
            col=1
        )

    fig.update_xaxes(title_text="Liczba epok", row=1, col=1)
    fig.update_xaxes(title_text="Liczba epok", row=2, col=1)
    fig.update_xaxes(title_text="Liczba epok", row=3, col=1)

    fig.update_yaxes(title_text="Binary Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="F1 Macro", row=2, col=1)
    fig.update_yaxes(title_text="Loss", row=3, col=1)

    fig.update_layout(
        width=1400,
        height=1200,
        title="Training Metrics"
    )

    fig.write_html(filename)


# ---------------------------------------------------------
# Seed - powtarzalność wyników
# ---------------------------------------------------------

np.random.seed(10)
keras.utils.set_random_seed(10)


# ---------------------------------------------------------
# Argumenty programu
# ---------------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument(
    "-i",
    "--images",
    required=True,
    help="ścieżka do katalogu z obrazami"
)

ap.add_argument(
    "-e",
    "--epochs",
    default=1,
    type=int,
    help="liczba epok"
)

ap.add_argument(
    "-m",
    "--model-type",
    default="mobilenet",
    choices=["mobilenet", "vgg"],
    help="typ modelu: 'mobilenet' (Transfer Learning) lub 'vgg' (ulepszona mała sieć z BN)"
)

args = vars(ap.parse_args())


# ---------------------------------------------------------
# Parametry treningu
# ---------------------------------------------------------

EPOCHS = args["epochs"]
LEARNING_RATE = 0.001
BATCH_SIZE = 32

INPUT_SHAPE = (
    150,
    150,
    3
)


# ---------------------------------------------------------
# Katalog wyjściowy
# ---------------------------------------------------------

os.makedirs(
    "output",
    exist_ok=True
)


# ---------------------------------------------------------
# Wczytywanie danych
# ---------------------------------------------------------

print("[INFO] Wczytywanie danych...")

image_paths = list(
    paths.list_images(
        args["images"]
    )
)

if len(image_paths) == 0:
    raise ValueError(
        f"Nie znaleziono obrazów w katalogu: {args['images']}"
    )

np.random.shuffle(
    image_paths
)

data = []
labels = []

for image_path in image_paths:
    image = cv2.imread(image_path)

    if image is None:
        print(f"[WARNING] Nie można wczytać: {image_path}")
        continue

    image = cv2.resize(
        image,
        (INPUT_SHAPE[1], INPUT_SHAPE[0])
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    data.append(image)

    parent_dir = os.path.basename(
        os.path.dirname(image_path)
    )

    label = parent_dir.split("_")
    labels.append(label)


# ---------------------------------------------------------
# Kontrola danych
# ---------------------------------------------------------

if len(data) == 0:
    raise ValueError(
        "Nie udało się wczytać żadnego obrazu."
    )


# ---------------------------------------------------------
# Konwersja do NumPy + normalizacja
# ---------------------------------------------------------

data = np.asarray(data, dtype="float32") / 255.0

print(f"[INFO] {len(data)} obrazów o rozmiarze: {data.nbytes / (1024 * 1024):.2f} MB")
print(f"[INFO] Kształt danych: {data.shape}")


# ---------------------------------------------------------
# Binaryzacja etykiet
# ---------------------------------------------------------

print("[INFO] Binaryzacja etykiet...")

mlb = MultiLabelBinarizer()
labels = mlb.fit_transform(labels)

print(f"[INFO] Etykiety: {mlb.classes_}")
print(f"[INFO] Liczba klas: {len(mlb.classes_)}")


# ---------------------------------------------------------
# Eksport MultiLabelBinarizer
# ---------------------------------------------------------

print("[INFO] Eksport etykiet do pliku...")

with open("output/mlb.pickle", "wb") as file:
    pickle.dump(mlb, file)


# ---------------------------------------------------------
# Podział Train / Test
# ---------------------------------------------------------

print("[INFO] Podział na zbiór treningowy i testowy...")

X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=10
)

print(f"[INFO] Rozmiar danych treningowych: {X_train.shape}")
print(f"[INFO] Rozmiar etykiet treningowych: {y_train.shape}")
print(f"[INFO] Rozmiar danych testowych: {X_test.shape}")
print(f"[INFO] Rozmiar etykiet testowych: {y_test.shape}")


# ---------------------------------------------------------
# Tworzenie pipeline'u tf.data.Dataset
# ---------------------------------------------------------

print("[INFO] Budowanie pipeline'u tf.data.Dataset...")

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=len(X_train)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------
# Augmentacja danych - Keras 3 (Poprawiona)
# ---------------------------------------------------------

print("[INFO] Budowa augmentacji...")

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomRotation(
            factor=30 / 360,
            fill_mode="nearest"
        ),
        keras.layers.RandomTranslation(
            height_factor=0.2,
            width_factor=0.2,
            fill_mode="nearest"
        ),
        keras.layers.RandomShear(
            x_factor=(0.0, 0.1),
            y_factor=(0.0, 0.1),
            fill_mode="nearest"
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.15, 0.15),
            width_factor=(-0.15, 0.15),
            fill_mode="nearest"
        ),
        keras.layers.RandomFlip("horizontal"),
    ],
    name="data_augmentation"
)


# ---------------------------------------------------------
# Budowa modelu
# ---------------------------------------------------------

print(f"[INFO] Budowa modelu (typ: {args['model_type']})...")

if args["model_type"] == "mobilenet":
    architecture = models.MobileNetV3Transfer(
        input_shape=INPUT_SHAPE,
        num_classes=len(mlb.classes_),
        final_activation="sigmoid",
        trainable=False
    )
else:
    architecture = models.VGGNetSmall(
        input_shape=INPUT_SHAPE,
        num_classes=len(mlb.classes_),
        final_activation="sigmoid"
    )

base_model = architecture.build()


# ---------------------------------------------------------
# Dodanie augmentacji przed właściwym siecią
# ---------------------------------------------------------

inputs = keras.Input(
    shape=INPUT_SHAPE,
    name="input_image"
)

x = data_augmentation(inputs)
outputs = base_model(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name=f"MultiLabel_{args['model_type']}"
)


# ---------------------------------------------------------
# Podsumowanie modelu
# ---------------------------------------------------------

model.summary()


# ---------------------------------------------------------
# Kompilacja
# ---------------------------------------------------------

print("[INFO] Kompilacja modelu...")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        keras.metrics.BinaryAccuracy(name="binary_accuracy", threshold=0.5),
        keras.metrics.Precision(name="precision", thresholds=0.5),
        keras.metrics.Recall(name="recall", thresholds=0.5),
        keras.metrics.F1Score(average="macro", threshold=0.5, name="f1_macro"),
        keras.metrics.F1Score(average="micro", threshold=0.5, name="f1_micro"),
    ]
)


# ---------------------------------------------------------
# Nazwa modelu i Checkpoint
# ---------------------------------------------------------

dt = datetime.now().strftime("%d_%m_%Y_%H_%M")
filepath = os.path.join("output", f"model_{dt}.keras")

checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor="val_f1_macro",
    mode="max",
    save_best_only=True,
    verbose=1
)


# ---------------------------------------------------------
# Trening
# ---------------------------------------------------------

print("[INFO] Trenowanie modelu...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint]
)


# ---------------------------------------------------------
# Wyniki końcowe
# ---------------------------------------------------------

print("[INFO] Trening zakończony.")
print(f"[INFO] Najlepszy model zapisany jako: {filepath}")


# ---------------------------------------------------------
# Raport HTML
# ---------------------------------------------------------

filename = os.path.join("output", f"report_{dt}.html")
print(f"[INFO] Eksport wykresu do pliku {filename}...")

plot_hist(history, filename=filename)

print("[INFO] Koniec")