from keras.models import load_model

import numpy as np
import pickle
import imutils
import argparse
import cv2
import os


# ---------------------------------------------------------
# Ograniczenie komunikatów TensorFlow
# ---------------------------------------------------------

import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ---------------------------------------------------------
# Parametry
# ---------------------------------------------------------

INPUT_SHAPE = (150, 150)

THRESHOLD = 0.5


# ---------------------------------------------------------
# Wczytanie i przygotowanie obrazu
# ---------------------------------------------------------

def load_image(filename):

    image = cv2.imread(filename)

    if image is None:
        raise ValueError(
            f"Nie można wczytać obrazu: {filename}"
        )

    # resize do rozmiaru używanego podczas treningu
    image = cv2.resize(
        image,
        INPUT_SHAPE
    )

    # Podczas treningu obrazy były konwertowane:
    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # float32 + normalizacja
    image = image.astype(
        "float32"
    ) / 255.0

    # Dodanie wymiaru batch:
    #
    # (150, 150, 3)
    # ->
    # (1, 150, 150, 3)

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ---------------------------------------------------------
# Argumenty programu
# ---------------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="ścieżka do obrazu"
)

ap.add_argument(
    "-m",
    "--model",
    required=True,
    help="ścieżka do modelu"
)

ap.add_argument(
    "-t",
    "--threshold",
    type=float,
    default=0.5,
    help='próg klasyfikacji etykiety, domyślnie jest to 0.5'
)

args = vars(
    ap.parse_args()
)

THRESHOLD = args["threshold"]

if not 0.0 <= THRESHOLD <= 1.0:
    raise ValueError(
        "Threshold musi znajdować się w zakresie 0.0 - 1.0"
    )


# ---------------------------------------------------------
# Ładowanie modelu
# ---------------------------------------------------------

print(
    "[INFO] Ładowanie modelu..."
)

model = load_model(
    args["model"],
    compile=False
)


# ---------------------------------------------------------
# Przygotowanie obrazu
# ---------------------------------------------------------

print(
    "[INFO] Przygotowanie obrazu..."
)

input_image = load_image(
    args["image"]
)


print(
    f"[INFO] Kształt wejścia: "
    f"{input_image.shape}"
)


# ---------------------------------------------------------
# Predykcja
# ---------------------------------------------------------

print(
    "[INFO] Klasyfikacja..."
)

y_pred = model.predict(
    input_image,
    verbose=0
)


# model zwraca:
#
# (1, liczba_klas)
#
# interesuje nas pierwszy obraz

y_pred = y_pred[0]


# ---------------------------------------------------------
# Ładowanie etykiet
# ---------------------------------------------------------

print(
    "[INFO] Ładowanie etykiet..."
)

with open(
    "output/mlb.pickle",
    "rb"
) as file:

    mlb = pickle.load(file)


labels = dict(
    enumerate(
        mlb.classes_
    )
)


# ---------------------------------------------------------
# Sortowanie predykcji
# ---------------------------------------------------------

# Klasy, które przekroczyły ustalony próg
idxs = np.where(
    y_pred >= THRESHOLD
)[0]

# Sortowanie tylko wybranych klas
# od największego prawdopodobieństwa
idxs = idxs[
    np.argsort(y_pred[idxs])[::-1]
]


# ---------------------------------------------------------
# Wyświetlenie wyników w konsoli
# ---------------------------------------------------------

print(
    f"[INFO] Wyniki dla threshold={THRESHOLD:.2f}:"
)

if len(idxs) == 0:

    print(
        "[INFO] Żadna etykieta nie przekroczyła "
        "ustalonego progu."
    )

    # Informacyjnie pokazujemy najlepszą predykcję
    best_idx = np.argmax(y_pred)

    print(
        f"[INFO] Najbardziej prawdopodobna klasa: "
        f"{labels[best_idx]} "
        f"({y_pred[best_idx] * 100:.2f}%)"
    )

else:

    for idx in idxs:

        print(
            f"    {labels[idx]:12s}: "
            f"{y_pred[idx] * 100:.2f}%"
        )


# ---------------------------------------------------------
# Wczytanie oryginalnego obrazu do prezentacji
# ---------------------------------------------------------

image = cv2.imread(
    args["image"]
)


if image is None:
    raise ValueError(
        f"Nie można wczytać obrazu: "
        f"{args['image']}"
    )


image = imutils.resize(
    image,
    width=1000
)


# ---------------------------------------------------------
# Dodanie dwóch najbardziej prawdopodobnych etykiet
# ---------------------------------------------------------

print("[INFO] Wyświetlanie obrazu...")

if len(idxs) > 0:

    for i, idx in enumerate(idxs):

        label = labels[idx]

        probability = (
            y_pred[idx] * 100
        )

        text = (
            f"{label}: "
            f"{probability:.2f}%"
        )

        cv2.putText(
            img=image,
            text=text,
            org=(
                10,
                (i * 35) + 30
            ),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.8,
            color=(
                0,
                179,
                137
            ),
            thickness=2
        )

else:

    best_idx = np.argmax(y_pred)

    text = (
        f"Brak klasy >= {THRESHOLD:.2f} "
        f"(best: {labels[best_idx]} "
        f"{y_pred[best_idx] * 100:.2f}%)"
    )

    cv2.putText(
        img=image,
        text=text,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 179, 137),
        thickness=2
    )

# ---------------------------------------------------------
# Wyświetlenie obrazu
# ---------------------------------------------------------

cv2.imshow(
    "Multi-label classification",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()