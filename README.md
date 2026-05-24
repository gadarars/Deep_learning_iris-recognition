# 🔍 Iris Recognition using Deep Learning

> Evolution from a custom CNN trained from scratch on limited data to state-of-the-art EfficientNet models scaled for real-world biometric identification.

**Course:** CSE488 (UG2023) — Deep Learning  
**University:** Ain Shams University, Faculty of Engineering  
**Authors:** Ahmed Lotfy (22p0251) · Roaa Sherif Gadara (22P0188)

---

## 📋 Overview

This project tackles the problem of iris-based person identification using deep learning, divided into two major phases of development:

**Phase 1 (Data-Scarce Environment - TensorFlow):**  
Given the small size of the MMU dataset (450 images), two approaches were initially compared:
1. **Simple CNN** — a custom convolutional network trained from scratch.
2. **MobileNetV2** — a lightweight pre-trained model fine-tuned via transfer learning.
*Finding: Transfer learning (83.33%) vastly outperformed training from scratch (15.56%).*

**Phase 2 (Production-Scale Environment - PyTorch):**  
To achieve state-of-the-art performance, the pipeline was rewritten in PyTorch and scaled up to the **CASIA-IrisV4 database** (18,851 images). Using a two-phase fine-tuning strategy, deeper architectures were deployed:
3. **ResNet-50** — a 50-layer deep residual network.
4. **EfficientNet-B0** — an optimized architecture scaling depth, width, and resolution.
*Finding: End-to-end fine-tuning on large datasets yields near-perfect accuracy (99.12%), completely solving the biometric classification challenge.*

---

## 📊 Results Summary

The evolution of our architectures yielded exceptional improvements across the board.

| Phase | Model | Training Strategy | Dataset | Classes | Validation Accuracy |
|---|---|---|---|---|---|
| 1 | Simple CNN | Scratch | MMU | 90 | 15.56% |
| 1 | MobileNetV2 | Transfer (Head Only) | MMU | 90 | 83.33% |
| 2 | ResNet-50 | Phase 2 (Fine-Tuned) | CASIA Combined | 820 | **98.99%** |
| 2 | **EfficientNet-B0** | **Phase 2 (Fine-Tuned)** | **CASIA Combined** | **820** | **99.12%** |

---

## 🗂️ Datasets

### 1. MMU (Multimedia University) Iris Dataset
*Used in Phase 1 for initial baseline testing.*
- **Total images:** 450
- **Subjects:** 45 (90 unique eyes/classes)
- **Format:** Greyscale bitmap (controlled conditions)

### 2. CASIA-IrisV4 (Interval + Lamp Combined)
*Used in Phase 2 for robust, production-grade scaling.*
- **Total images:** 18,851 (after filtering singletons)
- **Unique identities (Classes):** 820
- **Conditions:** Combined dataset forces the AI to learn biometric signatures invariant to changing illumination and non-linear pupil dilation.
- **Preprocessing:** Resized to `224×224`, normalized to ImageNet standards, stratified 80/20 train/validation split.

---

## 🏗️ Model Architectures & Strategies

### Model 1 — Simple CNN (Phase 1)
- 3 Conv2D blocks (32, 64, 128 filters) with MaxPooling and Batch Normalization.
- **2,214,234** parameters.

### Model 2 — MobileNetV2 (Phase 1)
- Frozen ImageNet base with a custom Dense (128) classification head.
- **175,578** trainable parameters.

### Model 3 & 4 — ResNet-50 and EfficientNet-B0 (Phase 2)
To maximize accuracy on the massive CASIA dataset, a **Two-Phase Fine-Tuning Strategy** was implemented:
1. **Phase 1 (Feature Extraction):** Base network is frozen. The custom classification head is trained for 10 epochs using Adam (LR = `1e-3`) to prevent gradient destruction.
2. **Phase 2 (End-to-End Fine-Tuning):** Entire network is unfrozen. A micro-learning rate (`1e-5`) is applied for 20 epochs, allowing deeper filters to adapt from recognizing ImageNet objects to microscopic iris crypts and furrows. 
- *Includes Smart Checkpointing to automatically save weights at peak validation accuracy.*

---

## 🛡️ Real-World Inference & Thresholding

To rigorously evaluate the system beyond the validation split, a simulated real-world deployment was conducted using the saved `efficientnet_b0_combined_best.pth` weights on novel, unseen data.

- **Intruder Rejection:** A **60% Confidence Threshold** was implemented. If the Softmax probability of the network's top prediction falls below 60%, the system flags the scan as an "Anomaly / Unknown" and denies access. This successfully prevents unregistered identities from forcing false positive matches.

---

## 🛠️ Tech Stack

- **Python 3.x**
- **TensorFlow 2.x / Keras** — Phase 1 modeling and data generators
- **PyTorch & Torchvision** — Phase 2 heavy architectures, dataloaders, and fine-tuning
- **Scikit-learn** — Stratified dataset splitting and advanced evaluation metrics
- **NumPy & Matplotlib** — Array operations and visualization
- **Tqdm** — Training loop progress tracking
- **Google Colab & Jupyter** — T4 GPU-accelerated training environment

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/Deep_learning_iris-recognition.git
cd Deep_learning_iris-recognition
```

### 2. Install dependencies
```bash
pip install tensorflow torch torchvision numpy matplotlib scikit-learn tqdm pillow
```

### 3. Download the datasets
- **Phase 1:** Download the [MMU Iris Dataset](https://www.kaggle.com/datasets/naureenmohammad/mmu-iris-dataset) from Kaggle.
- **Phase 2:** Request access to the [CASIA-IrisV4 Database](http://biometrics.idealtest.org/) and download the *Interval* and *Lamp* subsets. Place them in the `Data/` directory.

### 4. Run the notebooks
- `iris_recognition_TF.ipynb` for the Phase 1 TensorFlow baseline.
- `iris_extra_models_and_datasets.ipynb` for the Phase 2 PyTorch expansion, fine-tuning, and inference testing.

---

## 📁 Project Structure

```text
Deep_learning_iris-recognition/
├── Data/                      # (Add MMU and CASIA datasets here)
├── Test_Images/               # Unseen images for real-world inference testing
├── iris_recognition_TF.ipynb  # Phase 1 notebook (CNN & MobileNet)
├── iris_extra_models_and_datasets.ipynb # Phase 2 notebook (ResNet & EffNet)
├── efficientnet_b0_combined_best.pth    # Saved state-of-the-art model weights
├── README.md
└── report/
    └── Deep_Learning_iris.pdf # Full project report
```

---

## 🔮 Future Work

With the challenges of dataset scale, model architecture, and transfer learning successfully resolved in Phase 2, future work focuses on ultimate deployment challenges:

- **Iris Segmentation:** Integrate a U-Net model to explicitly isolate the iris region, cropping out eyelashes and eyelids to ensure the network cannot rely on periocular skin features.
- **Metric Learning (Siamese Networks):** Transition from standard classification (Softmax) to embedding distances (Triplet Loss), enabling the system to verify entirely new subjects without needing to retrain the final layer (Zero-Shot Learning).
- **Edge Deployment:** Convert the `.pth` PyTorch models to ONNX or TorchScript formats for deployment on embedded biometric hardware or mobile devices.

---

## 📚 References

1. Daugman, J. (2004). How iris recognition works. *IEEE TCSVT*, 14(1), 21–30.
2. Sandler et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. *CVPR*.
3. He, K. et al. (2016). Deep Residual Learning for Image Recognition. *CVPR*.
4. Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*.
5. Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. *NeurIPS*.
6. CASIA Iris Image Database (version 4.0). Institute of Automation, Chinese Academy of Sciences (CASIA).
