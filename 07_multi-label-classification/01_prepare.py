import os
from imutils import paths

BASE_IMG = os.path.join(os.getcwd(), 'downloads')

dir_paths = [os.path.join('downloads', dir) for dir in os.listdir(BASE_IMG)]
for dir_path in dir_paths:
    os.rename(dir_path, dir_path.replace(' ', '_'))

for dir_path in dir_paths:
    i = 0
    imgs = paths.list_images(dir_path)
    for img in imgs:
        print(img)

        directory = os.path.dirname(img)

        os.rename(img, os.path.join(directory, f'{i:04d}.jpg'))
        i += 1
