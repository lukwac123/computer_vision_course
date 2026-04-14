import os
import shutil
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, 'image-dataset')

CLS_1 = 'horse'
CLS_2 = 'lion'
TRAIN_RATIO = 0.7
VALID_RATIO = 0.2
DATA_DIR = os.path.join(BASE_DIR, 'images')

raw_no_of_files = {}
classes = [CLS_1, CLS_2]


def is_valid_image_file(filename):
    _, extension = os.path.splitext(filename)
    return extension.lower() in ['.jpg', '.png', '.jpeg']


def copy_split(files, source_dir, destination_dir):
    for filename in files:
        src = os.path.join(source_dir, filename)
        dst = os.path.join(destination_dir, filename)
        shutil.copyfile(src, dst)

number_of_samples = [(dir, len(os.listdir(os.path.join(DATASET_DIR, dir)))) for dir in classes]
print(number_of_samples)

os.makedirs(DATA_DIR, exist_ok=True)

# Katalogi do zbiorów: train, valid, test
train_dir = os.path.join(DATA_DIR, 'train')
valid_dir = os.path.join(DATA_DIR, 'valid')
test_dir = os.path.join(DATA_DIR, 'test')

train_cls_1_dir = os.path.join(train_dir, CLS_1)
valid_cls_1_dir = os.path.join(valid_dir, CLS_1)
test_cls_1_dir = os.path.join(test_dir, CLS_1)

train_cls_2_dir = os.path.join(train_dir, CLS_2)
valid_cls_2_dir = os.path.join(valid_dir, CLS_2)
test_cls_2_dir = os.path.join(test_dir, CLS_2)

for dir in (train_dir, valid_dir, test_dir):
    os.makedirs(dir, exist_ok=True)

for dir in (train_cls_1_dir, valid_cls_1_dir, test_cls_1_dir):
    os.makedirs(dir, exist_ok=True)

for dir in (train_cls_2_dir, valid_cls_2_dir, test_cls_2_dir):
    os.makedirs(dir, exist_ok=True)

print('[INFO] Wczytanie nazw plików...')
cls_1_names = os.listdir(os.path.join(DATASET_DIR, CLS_1))
cls_2_names = os.listdir(os.path.join(DATASET_DIR, CLS_2))

print('[INFO] Walidacja poprawności nazw...')
cls_1_names = [fname for fname in cls_1_names if is_valid_image_file(fname)]
cls_2_names = [fname for fname in cls_2_names if is_valid_image_file(fname)]

# Przetasowanie nazw plików
np.random.shuffle(cls_1_names)
np.random.shuffle(cls_2_names)

print(f'[INFO] Liczba obrazów w zbiorze {CLS_1}: {len(cls_1_names)}')
print(f'[INFO] Liczba obrazów w zbiorze {CLS_2}: {len(cls_2_names)}')

train_idx_cls_1 = int(TRAIN_RATIO * len(cls_1_names))
valid_idx_cls_1 = train_idx_cls_1 + int(VALID_RATIO * len(cls_1_names))

train_idx_cls_2 = int(TRAIN_RATIO * len(cls_2_names))
valid_idx_cls_2 = train_idx_cls_2 + int(VALID_RATIO * len(cls_2_names))

print('[INFO] Kopiowanie plików do katalogów docelowych...')
copy_split(cls_1_names[:train_idx_cls_1], os.path.join(DATASET_DIR, CLS_1), train_cls_1_dir)
copy_split(cls_1_names[train_idx_cls_1:valid_idx_cls_1], os.path.join(DATASET_DIR, CLS_1), valid_cls_1_dir)
copy_split(cls_1_names[valid_idx_cls_1:], os.path.join(DATASET_DIR, CLS_1), test_cls_1_dir)

copy_split(cls_2_names[:train_idx_cls_2], os.path.join(DATASET_DIR, CLS_2), train_cls_2_dir)
copy_split(cls_2_names[train_idx_cls_2:valid_idx_cls_2], os.path.join(DATASET_DIR, CLS_2), valid_cls_2_dir)
copy_split(cls_2_names[valid_idx_cls_2:], os.path.join(DATASET_DIR, CLS_2), test_cls_2_dir)

print(f'[INFO] Liczba obrazów klasy {CLS_1} w zbiorze treningowym: {len(os.listdir(train_cls_1_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_1} w zbiorze validacyjnym: {len(os.listdir(valid_cls_1_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_1} w zbiorze testowym: {len(os.listdir(test_cls_1_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_2} w zbiorze treningowym: {len(os.listdir(train_cls_2_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_2} w zbiorze validacyjnym: {len(os.listdir(valid_cls_2_dir))}')
print(f'[INFO] Liczba obrazów klasy {CLS_2} w zbiorze testowym: {len(os.listdir(test_cls_2_dir))}')
