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
# Keras
# ---------------------------------------------------------

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
        rows=2,
        cols=1,
        subplot_titles=(
            "Binary Accuracy",
            "Loss"
        )
    )

    # Accuracy - trening
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

    # Accuracy - walidacja
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

    # Loss - trening
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["loss"],
            name="train_loss",
            mode="markers+lines"
        ),
        row=2,
        col=1
    )

    # Loss - walidacja
    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["val_loss"],
            name="valid_loss",
            mode="markers+lines"
        ),
        row=2,
        col=1
    )

    fig.update_xaxes(
        title_text="Liczba epok",
        row=1,
        col=1
    )

    fig.update_xaxes(
        title_text="Liczba epok",
        row=2,
        col=1
    )

    fig.update_yaxes(
        title_text="Binary Accuracy",
        row=1,
        col=1
    )

    fig.update_yaxes(
        title_text="Loss",
        row=2,
        col=1
    )

    fig.update_layout(
        width=1400,
        height=1000,
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

    # Wczytanie obrazu
    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"[WARNING] Nie można wczytać: "
            f"{image_path}"
        )

        continue


    # Zmiana rozmiaru
    image = cv2.resize(
        image,
        (
            INPUT_SHAPE[1],
            INPUT_SHAPE[0]
        )
    )


    # OpenCV wczytuje obrazy jako BGR.
    # Konwertujemy do RGB.
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # Dodanie obrazu do danych
    data.append(
        image
    )


    # -----------------------------------------------------
    # Pobranie etykiet z nazwy katalogu
    #
    # przykład:
    #
    # downloads/black_trousers/0001.jpg
    #
    # katalog:
    # black_trousers
    #
    # etykiety:
    # ["black", "trousers"]
    # -----------------------------------------------------

    parent_dir = os.path.basename(
        os.path.dirname(
            image_path
        )
    )


    label = parent_dir.split(
        "_"
    )


    labels.append(
        label
    )


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

data = np.asarray(
    data,
    dtype="float32"
)


data = data / 255.0


print(
    f"[INFO] {len(data)} obrazów "
    f"o rozmiarze: "
    f"{data.nbytes / (1024 * 1024):.2f} MB"
)


print(
    f"[INFO] Kształt danych: "
    f"{data.shape}"
)


# ---------------------------------------------------------
# Binaryzacja etykiet
# ---------------------------------------------------------

print(
    "[INFO] Binaryzacja etykiet..."
)


mlb = MultiLabelBinarizer()


labels = mlb.fit_transform(
    labels
)


print(
    f"[INFO] Etykiety: "
    f"{mlb.classes_}"
)


print(
    f"[INFO] Liczba klas: "
    f"{len(mlb.classes_)}"
)


# ---------------------------------------------------------
# Eksport MultiLabelBinarizer
# ---------------------------------------------------------

print(
    "[INFO] Eksport etykiet do pliku..."
)


with open(
    "output/mlb.pickle",
    "wb"
) as file:

    pickle.dump(
        mlb,
        file
    )


# ---------------------------------------------------------
# Podział Train / Test
# ---------------------------------------------------------

print(
    "[INFO] Podział na zbiór treningowy "
    "i testowy..."
)


X_train, X_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    random_state=10
)


print(
    f"[INFO] Rozmiar danych treningowych: "
    f"{X_train.shape}"
)


print(
    f"[INFO] Rozmiar etykiet treningowych: "
    f"{y_train.shape}"
)


print(
    f"[INFO] Rozmiar danych testowych: "
    f"{X_test.shape}"
)


print(
    f"[INFO] Rozmiar etykiet testowych: "
    f"{y_test.shape}"
)


# ---------------------------------------------------------
# Augmentacja danych - Keras 3
# ---------------------------------------------------------

print(
    "[INFO] Budowa augmentacji..."
)


data_augmentation = keras.Sequential(
    [

        # Obrót do około +/- 30 stopni
        keras.layers.RandomRotation(
            factor=30 / 360,
            fill_mode="nearest"
        ),

        # Przesunięcie obrazu
        keras.layers.RandomTranslation(
            height_factor=0.2,
            width_factor=0.2,
            fill_mode="nearest"
        ),

        # Shear
        keras.layers.RandomShear(
            x_factor=np.tan(
                np.deg2rad(0.2)
            ),
            fill_mode="nearest"
        ),

        # Zoom
        keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest"
        ),

        # Odbicie poziome
        keras.layers.RandomFlip(
            "horizontal"
        ),

    ],
    name="data_augmentation"
)


# ---------------------------------------------------------
# Budowa modelu
# ---------------------------------------------------------

print(
    "[INFO] Budowa modelu..."
)


architecture = models.VGGNetSmall(
    input_shape=INPUT_SHAPE,
    num_classes=len(
        mlb.classes_
    ),
    final_activation="sigmoid"
)


base_model = architecture.build()


# ---------------------------------------------------------
# Dodanie augmentacji przed właściwą siecią CNN
# ---------------------------------------------------------

inputs = keras.Input(
    shape=INPUT_SHAPE,
    name="input_image"
)


x = data_augmentation(
    inputs
)


outputs = base_model(
    x
)


model = keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="VGGNetSmall_MultiLabel"
)


# ---------------------------------------------------------
# Podsumowanie modelu
# ---------------------------------------------------------

model.summary()


# ---------------------------------------------------------
# Kompilacja
# ---------------------------------------------------------

print(
    "[INFO] Kompilacja modelu..."
)


model.compile(

    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),

    loss="binary_crossentropy",

    metrics=[

        keras.metrics.BinaryAccuracy(
            name="binary_accuracy",
            threshold=0.5
        ),

        keras.metrics.Precision(
            name="precision",
            thresholds=0.5
        ),

        keras.metrics.Recall(
            name="recall",
            thresholds=0.5
        )

    ]
)


# ---------------------------------------------------------
# Nazwa modelu
# ---------------------------------------------------------

dt = datetime.now().strftime(
    "%d_%m_%Y_%H_%M"
)


filepath = os.path.join(
    "output",
    f"model_{dt}.keras"
)


# ---------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------

checkpoint = ModelCheckpoint(

    filepath=filepath,

    monitor="val_binary_accuracy",

    mode="max",

    save_best_only=True,

    verbose=1
)


# ---------------------------------------------------------
# Trening
# ---------------------------------------------------------

print(
    "[INFO] Trenowanie modelu..."
)


history = model.fit(

    X_train,
    y_train,

    validation_data=(
        X_test,
        y_test
    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[
        checkpoint
    ],

    shuffle=True
)


# ---------------------------------------------------------
# Wyniki końcowe
# ---------------------------------------------------------

print(
    "[INFO] Trening zakończony."
)


print(
    f"[INFO] Najlepszy model zapisany jako: "
    f"{filepath}"
)


# ---------------------------------------------------------
# Raport HTML
# ---------------------------------------------------------

filename = os.path.join(
    "output",
    f"report_{dt}.html"
)


print(
    f"[INFO] Eksport wykresu do pliku "
    f"{filename}..."
)


plot_hist(
    history,
    filename=filename
)


print(
    "[INFO] Koniec"
)