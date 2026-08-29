# Autism_Spectrum_Disorder
Detects autism spectrum traits from data using machine learning — includes data preprocessing, model training, and evaluation in a Jupyter notebook.

This repository contains the notebook used to detect autism spectrum traits using following Model Approach.

## Model Approach

Transfer learning with **InceptionV3** (pretrained on ImageNet) for binary image classification of autism spectrum traits from facial images.

- **Preprocessing:** Images resized to 299×299, with augmentation (rotation, shift, zoom, brightness, flip) applied to the training set
- **Architecture:** InceptionV3 base (frozen) → GlobalAveragePooling → BatchNormalization → Dense(256, ReLU, L2 regularization) → Dropout(0.5) → Dense(1, sigmoid)
- **Training strategy (two phases):**
  1. **Head training** — base model frozen, only top layers trained (Adam, lr=1e-3)
  2. **Fine-tuning** — last 150 layers of InceptionV3 unfrozen (BatchNorm layers kept frozen) and trained at a lower learning rate (1e-5)
- **Class imbalance handling:** Class weights computed via `compute_class_weight`
- **Callbacks:** EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
- **Evaluation:** Accuracy, classification report, and confusion matrix on the test set

## Files
- `autism_spectrum_improved__1_.ipynb` — main notebook with data preprocessing, model training, and evaluation

## Dataset
The dataset used in this project is available here:
[Google Drive Link]((https://drive.google.com/drive/folders/1X8F3JratusRCRbFfaAJzw2gj6SDDAVFV?usp=sharing))

## How to Run
1. Open the notebook in Google Colab or Kaggle
2. Download the dataset from the Drive link above
3. Update the dataset path in the notebook
4. Run all cells
