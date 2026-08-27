import cv2
import numpy as np
import os
import time
import argparse


# ---------------------------------------------------------
# Argumenty
# ---------------------------------------------------------

ap = argparse.ArgumentParser()

ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="path to image"
)

args = vars(ap.parse_args())


# ---------------------------------------------------------
# Parametry
# ---------------------------------------------------------

np.random.seed(10)

CONFIDENCE = 0.5
THRESHOLD = 0.3


# ---------------------------------------------------------
# Ścieżki YOLO
# ---------------------------------------------------------

print("[INFO] Loading labels...")

labels_path = os.path.join(
    "yolo",
    "coco.names"
)

weights_path = os.path.join(
    "yolo",
    "yolov3.weights"
)

config_path = os.path.join(
    "yolo",
    "yolov3.cfg"
)


# ---------------------------------------------------------
# Sprawdzenie plików
# ---------------------------------------------------------

for path in [
    labels_path,
    weights_path,
    config_path
]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Nie znaleziono pliku: {path}"
        )


# ---------------------------------------------------------
# Klasy COCO
# ---------------------------------------------------------

with open(labels_path, "r") as file:
    LABELS = file.read().strip().split("\n")


COLORS = np.random.randint(
    0,
    255,
    size=(len(LABELS), 3),
    dtype="uint8"
)


# ---------------------------------------------------------
# Ładowanie YOLO
# ---------------------------------------------------------

print("[INFO] Loading YOLO...")

net = cv2.dnn.readNetFromDarknet(
    config_path,
    weights_path
)


# ---------------------------------------------------------
# Warstwy wyjściowe
# ---------------------------------------------------------

ln = net.getUnconnectedOutLayersNames()

print(
    f"[INFO] Output layers: {ln}"
)


# ---------------------------------------------------------
# Wczytanie obrazu
# ---------------------------------------------------------

image = cv2.imread(
    args["image"]
)

if image is None:
    raise ValueError(
        f"Nie można wczytać obrazu: {args['image']}"
    )


(h, w) = image.shape[:2]


# ---------------------------------------------------------
# Blob
# ---------------------------------------------------------

blob = cv2.dnn.blobFromImage(
    image,
    scalefactor=1 / 255.0,
    size=(416, 416),
    swapRB=True,
    crop=False
)


net.setInput(blob)


# ---------------------------------------------------------
# Detekcja
# ---------------------------------------------------------

print("[INFO] Object detection...")

start = time.time()

layer_outputs = net.forward(
    ln
)

end = time.time()


print(
    f"[INFO] YOLO detection took "
    f"{end - start:.2f} seconds"
)


# ---------------------------------------------------------
# Wyniki
# ---------------------------------------------------------

boxes = []
confidences = []
class_ids = []


for output in layer_outputs:

    for detection in output:

        # prawdopodobieństwa klas
        scores = detection[5:]

        class_id = np.argmax(
            scores
        )

        # prawdopodobieństwo obiektu
        objectness = detection[4]

        # confidence klasy
        class_probability = scores[class_id]

        confidence = (
            objectness * class_probability
        )


        if confidence > CONFIDENCE:

            box = detection[0:4] * np.array(
                [w, h, w, h]
            )

            (
                x_center,
                y_center,
                width,
                height
            ) = box.astype("int")


            x = int(
                x_center - width / 2
            )

            y = int(
                y_center - height / 2
            )


            boxes.append(
                [
                    x,
                    y,
                    int(width),
                    int(height)
                ]
            )

            confidences.append(
                float(confidence)
            )

            class_ids.append(
                class_id
            )


# ---------------------------------------------------------
# Non-Maximum Suppression
# ---------------------------------------------------------

idxs = cv2.dnn.NMSBoxes(
    boxes,
    confidences,
    CONFIDENCE,
    THRESHOLD
)


# ---------------------------------------------------------
# Rysowanie wyników
# ---------------------------------------------------------

if len(idxs) > 0:

    for i in np.array(idxs).flatten():

        (x, y) = (
            boxes[i][0],
            boxes[i][1]
        )

        (box_w, box_h) = (
            boxes[i][2],
            boxes[i][3]
        )


        color = [
            int(c)
            for c in COLORS[class_ids[i]]
        ]


        cv2.rectangle(
            image,
            (x, y),
            (x + box_w, y + box_h),
            color,
            2
        )


        text = (
            f"{LABELS[class_ids[i]]}: "
            f"{confidences[i] * 100:.2f}%"
        )


        cv2.putText(
            image,
            text,
            (x, max(y - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )


# ---------------------------------------------------------
# Wyświetlenie
# ---------------------------------------------------------

cv2.imshow(
    "YOLOv3 Object Detection",
    image
)

cv2.waitKey(0)

cv2.destroyAllWindows()
