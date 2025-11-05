"""Predict defects from new images."""

import argparse
import os
from pathlib import Path
import numpy as np
import cv2
import joblib
import tensorflow as tf

from src.data_preprocessing import preprocess_image
from src.feature_extraction import extract_basic_stats, extract_gabor_features_gray


def extract_features_from_array(img_array):
    # expects image values in [0,1] or [0,255]
    img = img_array
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype('uint8')
    else:
        img_uint8 = img.astype('uint8')
    if img_uint8.ndim == 3 and img_uint8.shape[-1] == 3:
        img_gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_uint8 if img_uint8.ndim == 2 else img_uint8[...,0]
    feats = list(extract_basic_stats(img)) + list(extract_gabor_features_gray(img_gray))
    return np.array(feats).reshape(1, -1)


def predict_single_image(model_path, image_path, img_size=224):
    # choose by extension
    if Path(model_path).suffix in ['.pkl', '.joblib']:
        clf = joblib.load(model_path)
        img = preprocess_image(image_path, img_size=img_size)
        feats = extract_features_from_array(img)
        pred = clf.predict(feats)[0]
        probs = None
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(feats)[0]
        return pred, probs
    else:
        model = tf.keras.models.load_model(model_path)
        img = preprocess_image(image_path, img_size=img_size)
        x = np.expand_dims(img, 0)
        probs = model.predict(x)
        pred = np.argmax(probs, axis=1)[0]
        return pred, probs[0]


def predict_directory(model_path, dir_path, img_size=224):
    results = []
    for p in sorted(Path(dir_path).rglob('*')):
        if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            try:
                pred, probs = predict_single_image(model_path, str(p), img_size=img_size)
                results.append({'path': str(p), 'pred': int(pred), 'probs': None if probs is None else probs.tolist()})
            except Exception as e:
                results.append({'path': str(p), 'error': str(e)})
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True,
                        help='Path to an image file or a directory containing images')
    parser.add_argument('--img_size', type=int, default=224)
    args = parser.parse_args()

    model_path = args.model_path
    image_path = args.image_path

    if os.path.isdir(image_path):
        res = predict_directory(model_path, image_path, img_size=args.img_size)
        for r in res:
            print(r)
    else:
        pred, probs = predict_single_image(model_path, image_path, img_size=args.img_size)
        print('Pred:', pred)
        if probs is not None:
            print('Probs:', probs)
