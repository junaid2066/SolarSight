# SolarSight
SolarSight: An Intelligent Framework for Automated Detection of Surface Defects in Solar PV Panels. It is a machine learning–based framework designed to automatically detect and classify surface defects in photovoltaic (PV) solar panels.  
The project integrates both **classical computer vision** and **deep learning** methods to identify issues such as cracks, dust, and discoloration that can affect panel efficiency.

## Purpose
Train and evaluate machine learning models (classical and deep learning) for detecting defects in PV solar panel images. The original notebook uses scikit-learn, XGBoost, TensorFlow (Keras), OpenCV and scikit-image for preprocessing and modeling.

## 🧠 Overview

This framework provides a unified workflow for:
- Preprocessing and augmenting solar panel image datasets  
- Extracting meaningful visual features (texture, color, and structural cues)  
- Training and comparing **classical ML algorithms** (SVM, Random Forest, XGBoost) and **deep learning models** (CNNs)  
- Evaluating models with accuracy, precision, recall, F1-score, and confusion matrices  
- Deploying models for real-time or batch prediction on unseen PV panel images  

---

## 📊 Dataset

Public dataset used in this project:  
**PV Panel Defect Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/alicjalena/pv-panel-defect-dataset)

**Classes:**  
- Normal  
- Crack  
- Dust  
- Discoloration  

You can download and place the dataset inside the `data/` folder:


## Repo Structure
Files included below are ready to be copied into files in the repository. The suggested repo structure:
```bash
Solar-Defect-Detection/
├── data/ # dataset (not included) - instructions in README
├── models/ # saved models
├── notebooks/
│ └── Solar_Defect_Detection_In_Process.ipynb
├── src/
│ ├── data_preprocessing.py - dataset loading, image resizing, augmentation utilities, and classical feature extraction hooks.
│ ├── feature_extraction.py — implementations for Gabor features and other handcrafted descriptors used by classical ML methods.
│ ├── train_model.py — training orchestration for classical ML (XGBoost/RandomForest) and a TensorFlow CNN.
│ ├── evaluate_model.py — evaluation metrics and visualizations.
│ └── predict.py — load a saved model and run inference on single images or a directory.
├── requirements.txt
└── README.md
```

## Quickstart
1. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate # linux/mac
venv\Scripts\activate # windows
pip install -r requirements.txt
```
2. Place dataset into data/ as described above.

3. Preprocess & extract features (optional):
```bash
python src/data_preprocessing.py --data_dir data/ --out_dir data/processed

4. Train models:
```bash
python src/train_model.py --data_dir data/ --save_dir models/
```

5. Evaluate:
```bash
python src/evaluate_model.py --model_path models/best_model.pkl --test_dir data/test
```


6. Predict a single image:
```bash
python src/predict.py --model_path models/best_model.pkl --image_path path/to/image.jpg
```

## 📜 Citation

If you use this work or dataset in your research, please cite:

@misc{solarsight2025,

  author = {Muhammad Junaid Asif, Muhammad Saad, Usman Nazakat, Uzair Khan},
  
  title  = {SolarSight: An Intelligent Framework for Automated Detection of Surface Defects in Solar PV Panels},
  
  year   = {2025},
  
  publisher = {GitHub},
  
  url    = {https://github.com/junaid2066/SolarSight}
  
}

## 👨‍💻 Author

Muhammad Junaid Asif (AM-Tech)
Computer Vision and Artificial Intelligence Researcher
📧 junaid.asif@ncp.edu.pk
🌐 [[LinkedIn ](https://www.linkedin.com/in/mjunaid94ee/)
🌐 [[Portfolio]](https://sites.google.com/view/junaid94ee/about-me)






