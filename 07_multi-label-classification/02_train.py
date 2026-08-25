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

# suppress logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import keras
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from architecture import models


def plot_hist(history, filename):
    hist = pd.DataFrame(history.history)
    hist["epoch"] = history.epoch

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Accuracy", "Loss")
    )

    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["accuracy"],
            name="train_accuracy",
            mode="markers+lines"
        ),
        row=1,
        col=1
    )

    fig.add_trace(
        go.Scatter(
            x=hist["epoch"],
            y=hist["val_accuracy"],
            name="valid_accuracy",
            mode="markers+lines"
        ),
        row=1,
        col=1
    )

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

    fig.update_xaxes(title_text="Liczba epok", row=1, col=1)
    fig.update_xaxes(title_text="Liczba epok", row=2, col=1)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Loss", row=2, col=1)

    fig.update_layout(
        width=1400,
        height=1000,
        title="Metrics"
    )

    fig.write_html(filename)


np.random.seed(10)

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--images",
    required=True,
    help="path to the data"
)
ap.add_argument(
    "-e",
    "--epochs",
    default=1,
    type=int,
    help="number of epochs"
)

args = vars(ap.parse_args())

# Parametry
EPOCHS = args["epochs"]
LEARNING_RATE = 0.001
BATCH_SIZE = 32
INPUT_SHAPE = (150, 150, 3)

os.makedirs("output", exist_ok=True)

# Wczytywanie danych
print("[INFO] Wczytywanie danych...")

image_paths = list(paths.list_images(args["images"]))
np.random.shuffle(image_paths)

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

    # OpenCV używa BGR.
    # Konwertujemy do standardowego RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    data.append(image)

    # np.
    # downloads/black_trousers/0001.jpg
    #
    # parent_dir -> black_trousers

    parent_dir = os.path.basename(
        os.path.dirname(image_path)
    )

    label = parent_dir.split("_")

    labels.append(label)

# NumPy + normalizacja
data = np.asarray(data, dtype="float32") / 255.0

print(
    f"[INFO] {len(data)} obrazów o rozmiarze: "
    f"{data.nbytes / (1024 * 1024):.2f} MB"
)

print(f"[INFO] Kształt danych: {data.shape}")

# Binaryzacja etykiet
print("[INFO] Binaryzacja etykiet...")

mlb = MultiLabelBinarizer()

labels = mlb.fit_transform(labels)

print(f"[INFO] Etykiety: {mlb.classes_}")

# Eksport MultiLabelBinarizer
print("[INFO] Eksport etykiet do pliku...")

with open("output/mlb.pickle", "wb") as file:
    pickle.dump(mlb, file)

# Train / Test
print("[INFO] Podział na zbiór treningowy i testowy...")

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
    f"[INFO] Rozmiar danych testowych: "
    f"{X_test.shape}"
)

# Augmentacja Keras 3
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
            x_factor=np.tan(np.deg2rad(0.2)),
            fill_mode="nearest"
        ),

        keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="nearest"
        ),

        keras.layers.RandomFlip("horizontal"),
    ],
    name="data_augmentation"
)

# Model
print("[INFO] Budowa modelu...")

architecture = models.VGGNetSmall(
    input_shape=INPUT_SHAPE,
    num_classes=len(mlb.classes_),
    final_activation="sigmoid"
)

base_model = architecture.build()


# Augmentację umieszczamy przed właściwą siecią
inputs = keras.Input(shape=INPUT_SHAPE)

x = data_augmentation(inputs)

outputs = base_model(x)

model = keras.Model(
    inputs=inputs,
    outputs=outputs
)

model.summary()

# Kompilacja
model.compile(
    optimizer=Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Checkpoint
dt = datetime.now().strftime(
    "%d_%m_%Y_%H_%M"
)

filepath = os.path.join(
    "output",
    "model_" + dt + ".keras"
)

checkpoint = ModelCheckpoint(
    filepath=filepath,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True
)

# Trening
print("[INFO] Trenowanie modelu...")

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

# Raport
filename = os.path.join(
    "output",
    "report_" + dt + ".html"
)

print(
    f"[INFO] Eksport wykresu do pliku "
    f"{filename}..."
)

plot_hist(
    history,
    filename=filename
)

print("[INFO] Koniec")
