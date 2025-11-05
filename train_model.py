"""Train classical ML and CNN models."""

import os
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models


from src.data_preprocessing import build_dataset
from src.feature_extraction import extract_gabor_features_gray, extract_basic_stats
import cv2




def train_xgboost(X_images, y, save_path):
# Extract handcrafted features from images (convert to gray)
X_feats = []
for img in X_images:
img_gray = (img*255).astype('uint8')
if img_gray.ndim==3:
img_gray = cv2.cvtColor(img_gray, cv2.COLOR_RGB2GRAY)
feats = list(extract_basic_stats(img)) + list(extract_gabor_features_gray(img_gray))
X_feats.append(feats)
X_feats = np.vstack(X_feats)
clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100)
clf.fit(X_feats, y)
joblib.dump(clf, save_path)
return clf




def build_simple_cnn(input_shape=(224,224,3), n_classes=2):
model = models.Sequential([
layers.Input(shape=input_shape),
layers.Conv2D(32, 3, activation='relu'),
layers.MaxPool2D(),
layers.Conv2D(64, 3, activation='relu'),
layers.MaxPool2D(),
layers.Flatten(),
layers.Dense(128, activation='relu'),
layers.Dropout(0.5),
layers.Dense(n_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
return model




def train_cnn(X_train, y_train, X_val, y_val, save_dir, epochs=10, batch_size=32):
model = build_simple_cnn(input_shape=X_train.shape[1:], n_classes=len(np.unique(y_train)))
Path(save_dir).mkdir(parents=True, exist_ok=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
model.save(os.path.join(save_dir, 'cnn_model'))
return model, history




if __name__ == '__main__':
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--save_dir', type=str, default='models')
parser.add_argument('--img_size', type=int, default=224)
parser.add_argument('--epochs', type=int, default=10)
args = parser.parse_args()


ds = build_dataset(args.data_dir, out_dir=None, img_size=args.img_size)
X_train, X_test = ds['X_train'], ds['X_test']
y_train, y_test = ds['y_train'], ds['y_test']


Path(args.save_dir).mkdir(parents=True, exist_ok=True)


# Train XGBoost (classical)
print('Training XGBoost...')
xgb_path = os.path.join(args.save_dir, 'xgb_model.joblib')
xgb_clf = train_xgboost(X_train, y_train, xgb_path)


# Train simple CNN
print('Training CNN...')
cnn_model, history = train_cnn(X_train, y_train, X_test, y_test, args.save_dir, epochs=args.epochs)


print('Training complete. Models saved to', args.save_dir)