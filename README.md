# SolarSight
SolarSight: An Intelligent Framework for Automated Detection of Surface Defects in Solar PV Panels

## Purpose
Train and evaluate machine learning models (classical and deep learning) for detecting defects in PV solar panel images. The original notebook uses scikit-learn, XGBoost, TensorFlow (Keras), OpenCV and scikit-image for preprocessing and modeling.


## Dataset
Download the dataset from Kaggle:


https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset


Place the extracted dataset under the `data/` directory. Expected layout (example):

Adjust paths inside `src/config` variables or in the scripts if your dataset layout differs.


## Repo Structure
See the repository tree in the project root. Key scripts are under `src/`:


- `data_preprocessing.py` — dataset loading, image resizing, augmentation utilities, and classical feature extraction hooks.
- `feature_extraction.py` — implementations for Gabor features and other handcrafted descriptors used by classical ML methods.
- `train_model.py` — training orchestration for classical ML (XGBoost/RandomForest) and a TensorFlow CNN.
- `evaluate_model.py` — evaluation metrics and visualizations.
- `predict.py` — load a saved model and run inference on single images or a directory.


## Quickstart
1. Create a virtual environment and install dependencies:


```bash
python -m venv venv
source venv/bin/activate # linux/mac
venv\Scripts\activate # windows
pip install -r requirements.txt
