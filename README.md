<div align="center">

# 🔬 Medical Image Classification — Skin Cancer Detection Using Hybrid CNN-Transformer

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-006400?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Colab](https://img.shields.io/badge/Google_Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)

**An end-to-end deep learning pipeline for automated skin cancer classification using the HAM10000 dataset, leveraging CNN, VGG16 Transfer Learning, and a novel Hybrid CNN-Transformer architecture.**

[Overview](#-overview) •
[Dataset](#-dataset) •
[Methodology](#-methodology) •
[Models](#-models) •
[Results](#-results) •
[Installation](#-installation) •
[Team](#-team)

</div>

---

## 📋 Overview

Skin cancer is one of the most common forms of cancer worldwide. Early and accurate detection is critical for effective treatment. This project applies **deep learning** and **machine learning** techniques to classify dermatoscopic images of skin lesions into **7 diagnostic categories**, aiming to support dermatologists in clinical decision-making.

### Key Highlights

- 🧠 **Hybrid Architecture**: CNN backbone + Transformer encoder for enhanced feature extraction
- 📊 **Multiple Approaches**: Traditional ML, Transfer Learning (VGG16), and Hybrid CNN-Transformer
- 🎯 **PCA Dimensionality Reduction**: 256 → 36 features while retaining 95% variance
- 📈 **Best Accuracy**: ~79% with Random Forest on CNN-extracted + PCA-reduced features

---

## 📦 Dataset

The project uses the **HAM10000** (Human Against Machine with 10000 training images) dataset — a benchmark dataset for dermatoscopic image analysis.

| Property | Details |
|----------|---------|
| **Total Images** | 10,015 |
| **Classes** | 7 diagnostic categories |
| **Format** | 28×28 RGB (CSV pixel format) |
| **Source** | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |

### Diagnostic Categories

| Label | Diagnosis | Full Name |
|-------|-----------|-----------|
| 0 | akiec | Actinic Keratoses & Intraepithelial Carcinoma |
| 1 | bcc | Basal Cell Carcinoma |
| 2 | bkl | Benign Keratosis-like Lesions |
| 3 | df | Dermatofibroma |
| 4 | mel | Melanoma |
| 5 | nv | Melanocytic Nevi |
| 6 | vasc | Vascular Lesions |

---

## 🔬 Methodology

```mermaid
graph LR
    A["1. Data Loading"] --> B["2. Preprocessing"]
    B --> C["3. Feature Extraction"]
    C --> D["4. Dimensionality Reduction"]
    D --> E["5. Classification"]
    E --> F["6. Evaluation"]
    
    style A fill:#4CAF50,stroke:#333,color:#fff
    style B fill:#2196F3,stroke:#333,color:#fff
    style C fill:#FF9800,stroke:#333,color:#fff
    style D fill:#9C27B0,stroke:#333,color:#fff
    style E fill:#F44336,stroke:#333,color:#fff
    style F fill:#00BCD4,stroke:#333,color:#fff
```

### Pipeline Details

1. **Data Loading**: Import HAM10000 CSV (28×28 RGB pixel values)
2. **Preprocessing**: Reshape images, normalize pixel values (0–1), encode labels, stratified train/test split (80/20)
3. **Feature Extraction**: CNN backbone with Conv2D → BatchNorm → MaxPool → Dropout → GlobalAveragePooling → Dense(256)
4. **Dimensionality Reduction**: PCA with 95% variance retention (256 → 36 components)
5. **Classification**: Random Forest & XGBoost with RandomizedSearchCV hyperparameter tuning
6. **Evaluation**: Accuracy, Classification Report, Confusion Matrix, Training curves

---

## 🧠 Models

### Model 1: Traditional ML Approach
> `Skin_Cancer_ML_Approach_Final.ipynb`

Initial exploration with data analysis, visualization, and baseline ML models on the HAM10000 dataset.

### Model 2: VGG16 Transfer Learning
> `CNN_VGG16.ipynb`

Pre-trained **VGG16** backbone with fine-tuning for skin lesion classification. Images resized to 128×128 for VGG16 compatibility.

### Model 3: Hybrid CNN-Transformer (Best Model)
> `BestHybird.ipynb`

A novel hybrid architecture combining:

| Component | Architecture | Purpose |
|-----------|-------------|---------|
| **CNN Backbone** | Conv2D(32) → Conv2D(64) → Conv2D(128) → GAP | Spatial feature extraction |
| **Transformer Encoder** | MultiHeadAttention (2 heads, key_dim=32) + LayerNorm | Capture complex feature relationships |
| **Fusion Layer** | Concatenate(CNN, Transformer) → Dense(128) → Dropout(0.4) | Combined feature classification |
| **Traditional ML** | PCA(256→36) + RandomForest / XGBoost | Enhanced classification with tuned hyperparameters |

---

## 📊 Results

### CNN Feature Extractor + Traditional ML

| Model | Accuracy | Macro Avg F1 | Weighted Avg F1 |
|-------|----------|-------------|----------------|
| **Random Forest** | **78.98%** | 0.59 | 0.78 |
| **XGBoost** | 78.53% | 0.59 | 0.77 |

### Hybrid CNN-Transformer (End-to-End)

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 77.33% |
| **Training Epochs** | 100 |
| **Batch Size** | 32 |

### Hyperparameter Tuning (Best Parameters)

| Model | n_estimators | max_depth | Other |
|-------|-------------|-----------|-------|
| Random Forest | 100–300 | 10–20, None | min_samples_split: 2–10 |
| XGBoost | 100–300 | 3–9 | learning_rate: 0.01–0.2 |

---

## 🚀 Installation

### Prerequisites

- Python 3.10+
- GPU recommended (CUDA-enabled)

### Setup

```bash
# Clone the repository
git clone https://github.com/KHALEDRABBAH/Medical-Image-Classification.git
cd Medical-Image-Classification

# Install dependencies
pip install tensorflow scikit-learn xgboost pandas numpy matplotlib joblib

# Or run directly in Google Colab (recommended)
```

### Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `tensorflow` | ≥2.x | Deep learning framework |
| `scikit-learn` | Latest | ML algorithms, PCA, metrics |
| `xgboost` | Latest | Gradient boosting classifier |
| `pandas` | Latest | Data manipulation |
| `numpy` | Latest | Numerical computing |
| `matplotlib` | Latest | Visualization |
| `joblib` | Latest | Model serialization |

---

## 📁 Project Structure

```
Medical-Image-Classification/
├── Skin_Cancer_ML_Approach_Final.ipynb    # Traditional ML approach
├── CNN_VGG16.ipynb                        # VGG16 transfer learning
├── BestHybird.ipynb                       # Hybrid CNN-Transformer (best)
├── README.md                              # Project documentation
├── LICENSE                                # MIT License
└── .gitignore                             # Git ignore rules
```

---

## 🔮 Future Work

- [ ] Implement data augmentation (rotation, flipping, zoom) to improve generalization
- [ ] Experiment with deeper Transformer architectures (ViT, DeiT)
- [ ] Add class weighting to address dataset imbalance
- [ ] Build a Streamlit/Gradio web app for real-time prediction
- [ ] Train on higher resolution images (224×224)
- [ ] Explore ensemble methods combining all three approaches

---

## 👥 Team

<div align="center">

| Team Member |
|-------------|
| **Khaled Rabbah** |
| **Habiba Samy** |

</div>

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🔗 Connect

<div align="center">

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Khaled_Rabbah-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/khaled-rabbah/)
[![GitHub](https://img.shields.io/badge/GitHub-KHALEDRABBAH-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/KHALEDRABBAH)

</div>

---

<div align="center">
<i>If you find this project useful, please consider giving it a ⭐</i>
</div>
