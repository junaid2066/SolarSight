"""Evaluate trained models on test data."""

"""Evaluate saved models and produce reports/plots."""
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import tensorflow as tf
from src.data_preprocessing import build_dataset


def evaluate_classical(model_path, X_test, y_test, feature_extractor=None):
    clf = joblib.load(model_path)
    # If the model expects features, we use feature_extractor if provided
    if feature_extractor is not None:
        X_feats = feature_extractor(X_test)
        y_pred = clf.predict(X_feats)
    else:
        try:
            y_pred = clf.predict(X_test)
        except Exception as e:
            raise RuntimeError('Model prediction failed. Provide a feature_extractor to convert raw images to features.')
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    return y_test, y_pred


def evaluate_cnn(model_path, X_test, y_test, batch_size=32):
    model = tf.keras.models.load_model(model_path)
    y_probs = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_probs, axis=1)
    print(classification_report(y_test, y_pred))
    print('Accuracy:', accuracy_score(y_test, y_pred))
    return y_test, y_pred


def plot_confusion(y_true, y_pred, labels=None, out_path=None):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_title('Confusion Matrix')
    n = len(cm)
    if labels is None:
        labels = list(range(n))
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, cm[i, j], ha='center', va='center')
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path)
        print(f'Saved confusion matrix to {out_path}')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--out_dir', type=str, default='reports')
    args = parser.parse_args()

    ds = build_dataset(args.data_dir, out_dir=None, img_size=args.img_size)
    X_test = ds['X_test']
    y_test = ds['y_test']
    label_map = ds.get('label_map', None)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    model_path = args.model_path
    # Decide type by presence of file extension
    if Path(model_path).suffix in ['.pkl', '.joblib']:
        # For classical models we don't have a generic feature extractor here.
        print('Evaluating classical model...')
        y_true, y_pred = evaluate_classical(model_path, X_test, y_test)
    else:
        print('Evaluating CNN model...')
        y_true, y_pred = evaluate_cnn(model_path, X_test, y_test)

    labels = None
    if label_map is not None:
        inv_map = {v:k for k,v in label_map.items()}
        labels = [inv_map[i] for i in sorted(inv_map.keys())]

    plot_confusion(y_true, y_pred, labels=labels, out_path=os.path.join(args.out_dir, 'confusion_matrix.png'))
