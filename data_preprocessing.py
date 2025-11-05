"""Data preprocessing utilities for Solar Defect Detection."""


import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import joblib




def load_image_paths(data_dir, exts={'.jpg','.jpeg','.png'}):
data_dir = Path(data_dir)
items = []
for cls in data_dir.iterdir():
if cls.is_dir():
for p in cls.rglob('*'):
if p.suffix.lower() in exts:
items.append((str(p), cls.name))
return items


def preprocess_image(img_path, img_size=224, gray=False):
img = cv2.imread(img_path)
if img is None:
raise ValueError(f"Could not read image: {img_path}")
if gray:
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (img_size, img_size))
img = img[..., np.newaxis]
else:
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (img_size, img_size))
img = img.astype('float32') / 255.0
return img




def build_dataset(data_dir, out_dir=None, img_size=224, test_size=0.2, random_state=42):
items = load_image_paths(data_dir)
paths, labels = zip(*items)
unique_labels = sorted(list(set(labels)))
label_map = {l:i for i,l in enumerate(unique_labels)}
y = [label_map[l] for l in labels]


X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
paths, y, paths, test_size=test_size, random_state=random_state, stratify=y)


def process_list(paths_list):
images = []
for p in tqdm(paths_list):
images.append(preprocess_image(p, img_size))
return np.stack(images, axis=0)


Xtr = process_list(X_train)
Xte = process_list(X_test)


out = {}
out['X_train'] = Xtr
out['X_test'] = Xte
out['y_train'] = np.array(y_train)
out['y_test'] = np.array(y_test)
out['label_map'] = label_map


if out_dir:
Path(out_dir).mkdir(parents=True, exist_ok=True)
joblib.dump(out, os.path.join(out_dir, 'processed_dataset.pkl'))
print(f"Saved processed dataset to {out_dir}/processed_dataset.pkl")


return out




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--out_dir', type=str, default='data/processed')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--test_size', type=float, default=0.2)
args = parser.parse_args()
build_dataset(args.data_dir, args.out_dir, img_size=args.img_size, test_size=args.test_size)