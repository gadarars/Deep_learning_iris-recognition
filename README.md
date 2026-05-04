# 🔍 Iris Recognition using Convolutional Neural Networks

> Comparing a custom CNN trained from scratch vs. MobileNetV2 Transfer Learning for biometric iris identification.

**Course:** CSE488 (UG2023) — Deep Learning  
**University:** Ain Shams University, Faculty of Engineering  
**Authors:** Ahmed Lotfy (22p0251) · Roaa Sherif Gadara (22P0188)

---

## 📋 Overview

This project tackles the problem of iris-based person identification using deep learning. Given the small size of the available dataset, two approaches are compared:

1. **Simple CNN** — a custom convolutional network trained from scratch.
2. **MobileNetV2** — a lightweight pre-trained model fine-tuned via transfer learning.

The key finding: transfer learning achieves **83.33%** validation accuracy vs. **15.56%** for the CNN from scratch — a **+67.78 percentage point** improvement.

---

## 📊 Results

| Model | Validation Accuracy | Validation Loss |
|---|---|---|
| Simple CNN (from scratch) | 15.56% | 3.9545 |
| MobileNetV2 (Transfer Learning) | **83.33%** | **1.3570** |
| **Improvement** | **+67.78%** | — |

---

## 🗂️ Dataset

**MMU (Multimedia University) Iris Dataset** — sourced from [Kaggle](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset).

| Property | Value |
|---|---|
| Total images | 450 |
| Subjects | 45 |
| Eyes per subject | 2 (left & right) |
| Images per eye | 5 |
| Total classes | 90 |
| Format | Greyscale bitmap |

**Preprocessing applied:**
- Images resized to `64×64` (CNN) and `224×224` (MobileNetV2)
- Pixel values normalised to `[0, 1]`
- Mild augmentation: rotation ±10°, zoom ±10%
- 80/20 train/validation split (360 train · 90 val)

---

## 🏗️ Model Architectures

### Model 1 — Simple CNN

| Layer | Output Shape | Parameters |
|---|---|---|
| Conv2D (32 filters, 3×3) + BN | (64, 64, 32) | 896 |
| MaxPooling2D (2×2) | (32, 32, 32) | 0 |
| Conv2D (64 filters, 3×3) + BN | (32, 32, 64) | 18,496 |
| MaxPooling2D (2×2) | (16, 16, 64) | 0 |
| Conv2D (128 filters, 3×3) + BN | (16, 16, 128) | 73,856 |
| MaxPooling2D (2×2) | (8, 8, 128) | 0 |
| Flatten | (8192,) | 0 |
| Dense (256) + Dropout (0.4) | (256,) | 2,097,408 |
| Dense (90, Softmax) | (90,) | 23,130 |
| **Total** | | **2,214,234** |

### Model 2 — MobileNetV2 Transfer Learning

| Component | Details | Status |
|---|---|---|
| MobileNetV2 Base | 154 layers, pre-trained on ImageNet | Frozen |
| GlobalAveragePooling2D | Compresses feature maps | Trainable |
| Dense (128, ReLU) | Custom classification head | Trainable |
| Dropout (0.3) | Regularisation | Trainable |
| Dense (90, Softmax) | Output — 90 identity classes | Trainable |
| **Total Trainable Params** | | **175,578** |

---

## ⚙️ Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimiser | Adam |
| Loss | Categorical Cross-Entropy |
| Metric | Accuracy |
| Max epochs | 15 |
| Batch size | 16 |
| Early stopping | Patience = 4 (monitors val accuracy) |

---

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow 2.x / Keras** — model definition, training, evaluation
- **NumPy** — array operations
- **Matplotlib** — training history plots and prediction visualisations
- **Google Colab** — T4 GPU-accelerated training environment

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/iris-recognition-cnn.git
cd iris-recognition-cnn
```

### 2. Install dependencies
```bash
pip install tensorflow numpy matplotlib
```

### 3. Download the dataset
Download the [MMU Iris Dataset from Kaggle](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset) and place it in the project root, or update the dataset path in the notebook.

### 4. Run the notebook
Open and run `iris_recognition.ipynb` in Google Colab (recommended for GPU access) or locally with Jupyter.

---

## 📁 Project Structure

```
iris-recognition-cnn/
├── iris_recognition.ipynb     # Main training notebook
├── README.md
└── report/
    └── Deep_Learning_iris.pdf # Full project report
```

---

## 🔮 Future Work

- **Fine-tuning** — unfreeze top MobileNetV2 layers for end-to-end training
- **Larger datasets** — evaluate on CASIA-IrisV4 or IITD Iris Database
- **Iris segmentation** — explicitly isolate the iris region before classification
- **Other architectures** — compare with EfficientNet, ResNet-50, Vision Transformers
- **Edge deployment** — convert to TensorFlow Lite for mobile/embedded devices

---

## 📚 References

1. Daugman, J. (2004). How iris recognition works. *IEEE TCSVT*, 14(1), 21–30.
2. Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
3. Chollet, F. et al. (2015). [Keras](https://keras.io).
4. Abadi, M. et al. (2016). TensorFlow: A system for large-scale machine learning. *OSDI*.
