from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
import argparse
import pickle
import os

# suppress logs
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from keras.models import load_model

# przykładowe uruchomienie
# $ python 03_predict.py -d test -m output/model_14_04_2026_14_17.keras

ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='type of images: [train, valid, test]')
ap.add_argument('-m', '--model', required=False, help='path to model')
args = vars(ap.parse_args())

INPUT_SHAPE = (150, 150, 3)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')


def resolve_dataset_path(dataset_arg):
    if os.path.isabs(dataset_arg):
        return dataset_arg

    candidate_paths = [
        os.path.join(BASE_DIR, dataset_arg),
        os.path.join(BASE_DIR, 'images', dataset_arg)
    ]

    for candidate in candidate_paths:
        if os.path.isdir(candidate):
            return candidate

    return os.path.join(BASE_DIR, dataset_arg)


def resolve_model_path(model_arg):
    if model_arg:
        return model_arg if os.path.isabs(model_arg) else os.path.join(BASE_DIR, model_arg)

    model_files = [
        os.path.join(OUTPUT_DIR, filename)
        for filename in os.listdir(OUTPUT_DIR)
        if filename.endswith('.keras')
    ]

    if not model_files:
        raise FileNotFoundError('Nie znaleziono żadnego modelu .keras w katalogu output.')

    return max(model_files, key=os.path.getmtime)


def load_label_map(default_map):
    labels_path = os.path.join(OUTPUT_DIR, 'labels.pickle')

    if not os.path.exists(labels_path):
        return default_map

    with open(labels_path, 'rb') as file:
        return pickle.load(file)

dataset_dir = resolve_dataset_path(args['dataset'])
model_path = resolve_model_path(args['model'])

datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1. / 255.
)

generator = datagen.flow_from_directory(
    directory=dataset_dir,
    target_size=INPUT_SHAPE[:2],
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

print('[INFO] Wczytywanie modelu...')
model = load_model(model_path)

y_prob = model.predict(generator)
y_prob = y_prob.ravel()

y_true = generator.classes

predictions = pd.DataFrame({'y_prob': y_prob, 'y_true': y_true}, index=generator.filenames)
predictions['y_pred'] = predictions['y_prob'].apply(lambda x: 1 if x > 0.5 else 0)
predictions['is_incorrect'] = (predictions['y_true'] != predictions['y_pred']) * 1
errors = list(predictions[predictions['is_incorrect'] == 1].index)
print(predictions.head())

y_pred = predictions['y_pred'].values

print(f'[INFO] Macierz konfuzji:\n{confusion_matrix(y_true, y_pred)}')
label_map = load_label_map(generator.class_indices)
target_names = [label for label, _ in sorted(label_map.items(), key=lambda item: item[1])]
print(f'[INFO] Raport klasyfikacji:\n{classification_report(y_true, y_pred, target_names=target_names)}')
print(f'[INFO] Dokładność modelu: {accuracy_score(y_true, y_pred) * 100:.2f}%')

label_map = dict((v, k) for k, v in label_map.items())
predictions['class'] = predictions['y_pred'].apply(lambda x: label_map[x])

os.makedirs(OUTPUT_DIR, exist_ok=True)
predictions.to_csv(os.path.join(OUTPUT_DIR, 'predictions.csv'))

print(f'[INFO] Błędnie sklasyfikowano: {len(errors)}\n[INFO] Nazwy plików:')
for error in errors:
    print(error)