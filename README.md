# Rice-Leaf-Disease-Detection
Rice leaf disease classification using MobileNetV2, EfficientNetB0 and SVM.
# Rice Leaf Disease Detection Using Deep Learning and Transfer Learning

## 1. Project Overview

This project focuses on classifying rice leaf images into three disease categories using computer vision, transfer learning, and machine learning.

The objective is to analyze the dataset, experiment with different image classification approaches, evaluate data augmentation techniques, compare model performance, and identify a suitable final model candidate.

The project uses MobileNetV2 pretrained on ImageNet for feature extraction and a Linear Support Vector Machine (SVM) for classification.

## 2. Objectives

- Perform exploratory data analysis (EDA) on rice leaf images.
- Identify dataset quality issues and class distribution.
- Build a baseline CNN from scratch.
- Experiment with transfer learning using MobileNetV2 and EfficientNetB0.
- Evaluate the impact of image augmentation.
- Compare multiple classification models using stratified 5-fold cross-validation.
- Analyze classification performance using accuracy, macro F1-score, classification reports, and confusion matrices.
- Select and save a final model candidate for inference.

## 3. Dataset

The dataset contains three rice leaf disease categories:

| Disease Class | Number of Images |
|---|---:|
| Bacterial Leaf Blight | 40 |
| Brown Spot | 40 |
| Leaf Smut | 39 |
| **Total** | **119** |

**Dataset discrepancy:** The project specification mentions 120 images, but the supplied dataset contains 119 valid images. No additional image was artificially created to compensate for this difference.

The dataset contains RGB JPEG images with varying resolutions and aspect ratios.

The original dataset is not included in this repository. Refer to `data/README.md` for dataset information.

## 4. Exploratory Data Analysis

The EDA examined:

- Class distribution and image counts.
- Image dimensions and aspect ratios.
- Image formats and file sizes.
- Brightness and RGB channel statistics.
- Image sharpness using Laplacian variance.
- Exact duplicate images using MD5 hashes.
- Corrupted or unreadable image files.

### Key Findings

- The dataset contains 119 valid images across three classes.
- No exact duplicate images were detected.
- No corrupted images were detected.
- Image dimensions and aspect ratios vary considerably.
- Differences in brightness and image sharpness were observed across samples.

All valid images were retained. Low Laplacian variance was not treated as sufficient evidence to discard an image because it can also reflect image resolution and natural texture differences.

## 5. Image Preprocessing

Images were resized while preserving their original aspect ratios.

| Parameter | Value |
|---|---|
| Target image size | 128 × 448 |
| Color format | RGB |
| Data type | float32 |
| Pixel range | [0, 1] |
| Resize method | Aspect-ratio-preserving resize with padding |

The rectangular input shape was selected after inspecting the dataset's predominantly wide landscape images.

A direct-resizing experiment was also conducted to compare the effect of distorting image proportions.

## 6. Models and Experiments

The following approaches were evaluated:

1. Custom CNN trained from scratch.
2. Custom CNN with data augmentation.
3. MobileNetV2 with a trainable classification head.
4. MobileNetV2 feature extraction with Linear SVM.
5. MobileNetV2 feature extraction with RBF SVM.
6. EfficientNetB0 feature extraction with Linear SVM.
7. EfficientNetB0 feature extraction with RBF SVM.
8. MobileNetV2 with data augmentation and Linear SVM.
9. MobileNetV2 and EfficientNetB0 feature fusion with Linear SVM.
10. PCA combined with MobileNetV2 features and Linear SVM.

The transfer-learning models used ImageNet-pretrained feature extractors.

## 7. Model Comparison

Models were evaluated using stratified 5-fold cross-validation with consistent outer folds.

| Model | Mean CV Accuracy | Mean CV Macro F1 |
|---|---:|---:|
| MobileNetV2 + EfficientNetB0 Fusion | 92.32% | 92.19% |
| MobileNetV2 + Linear SVM | 92.32% | 92.05% |
| EfficientNetB0 + RBF SVM | 91.52% | 91.48% |
| EfficientNetB0 + Linear SVM | 91.49% | 91.35% |
| MobileNetV2 + RBF SVM | 91.49% | 91.32% |
| MobileNetV2 + Augmentation + Linear SVM | 88.88% | 87.38% |
| Custom CNN | 61.38% | 59.97% |
| Custom CNN + Augmentation | 55.36% | 52.65% |
| MobileNetV2 + Dense Classification Head | 40.40% | 37.52% |

*Values represent mean outer-fold cross-validation results. They are not independent external test-set results.*

## 8. Final Model Selection

**Selected candidate: MobileNetV2 Feature Extractor + Linear SVM**

Reasons for selection:

- Achieved 92.32% mean cross-validation accuracy.
- Achieved 92.05% mean cross-validation macro F1-score.
- Matched the feature-fusion model's mean cross-validation accuracy.
- Had only a small macro F1 difference compared with feature fusion.
- Uses fewer extracted features than the fusion approach, reducing feature storage and processing overhead.

The final SVM was trained on all 119 available images using C = 0.01.

The final model is a candidate for further validation, not a production-validated agricultural diagnostic system.

## 9. Out-of-Fold Evaluation

The selected MobileNetV2 + Linear SVM model achieved:

| Metric | Result |
|---|---:|
| OOF Accuracy | 92.44% |
| OOF Macro F1-score | 92.38% |
| Correct Predictions | 110 |
| Incorrect Predictions | 9 |

The out-of-fold predictions were generated so that each image was evaluated by a model that did not train on that image.

The confusion matrix and classification report are available in the `results/` directory.

## 10. Data Augmentation Analysis

Data augmentation was evaluated separately for the custom CNN and transfer-learning pipeline.

Techniques included:

- Horizontal flipping.
- Small rotations.
- Small zoom transformations.

In the tested configurations, augmentation did not improve performance.

| Experiment | Mean CV Accuracy | Mean CV Macro F1 |
|---|---:|---:|
| Custom CNN | 61.38% | 59.97% |
| Custom CNN + Augmentation | 55.36% | 52.65% |
| MobileNetV2 + Linear SVM | 92.32% | 92.05% |
| MobileNetV2 + Augmentation + Linear SVM | 88.88% | 87.38% |

These results apply to the tested augmentation settings and this small dataset. They do not establish that augmentation is generally ineffective for rice leaf disease classification.

Detailed results are available in `results/rice_leaf_augmentation_analysis.csv`.

## 11. Error Analysis

The selected model produced nine incorrect out-of-fold predictions.

The misclassified images were visually inspected. Some images appeared similar to examples from their predicted classes, suggesting that overlapping visible symptoms may contribute to classification difficulty.

This is a qualitative observation, not a confirmed explanation of the errors.

## 12. Saved Model Artifacts

The `models/` directory contains:

- `rice_leaf_final_svm.joblib` — trained Linear SVM classifier.
- `rice_leaf_mobilenet_feature_extractor.keras` — pretrained MobileNetV2 feature extractor.
- `rice_leaf_label_encoder.joblib` — class label encoder.

The inference pipeline resizes an input image, extracts MobileNetV2 features, predicts its class using the SVM, and converts the predicted label into its disease category.

## 13. Repository Structure

```text
Rice-Leaf-Disease-Detection/
├── data/
│   └── README.md
├── models/
│   ├── rice_leaf_final_svm.joblib
│   ├── rice_leaf_mobilenet_feature_extractor.keras
│   └── rice_leaf_label_encoder.joblib
├── results/
│   ├── rice_leaf_augmentation_analysis.csv
│   ├── rice_leaf_classification_report.csv
│   ├── rice_leaf_confusion_matrix.png
│   ├── rice_leaf_model_comparison.csv
│   └── rice_leaf_project_summary.csv
├── notebooks/
│   └── rice_leaf_disease_detection.ipynb
├── README.md
└── requirements.txt
```

## 14. Installation and Execution

### Clone the repository

```bash
git clone https://github.com/PAVANREDDYCEASE/Rice-Leaf-Disease-Detection
cd Rice-Leaf-Disease-Detection
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

```bash
jupyter notebook
```

Open:

`notebooks/rice_leaf_disease_detection.ipynb`

The original dataset must be obtained separately and placed in the expected directory structure before rerunning the complete notebook.

## 15. Limitations

- The dataset is very small, containing only 119 images.
- There is no independent external test dataset.
- Cross-validation results may not represent performance on images from new farms, cameras, regions, or environmental conditions.
- Repeated experimentation and model selection on the same dataset can introduce selection optimism.
- The model has not been validated by agricultural experts or tested in real-world field conditions.
- Confidence estimates should not be interpreted as verified disease probabilities without appropriate calibration and validation.

## 16. Future Improvements

- Collect a larger and more diverse dataset.
- Obtain independent test images from different agricultural environments.
- Evaluate performance across different cameras, lighting conditions, and rice varieties.
- Conduct external validation with agricultural experts.
- Investigate model calibration and uncertainty estimation.
- Explore explainability methods such as Grad-CAM.
- Evaluate the model on real-world field images before considering deployment.

## 17. Conclusion

This project compared custom CNNs, transfer-learning approaches, SVM classifiers, data augmentation, PCA, and feature fusion for rice leaf disease classification.

MobileNetV2 feature extraction combined with a Linear SVM was selected as the final model candidate, achieving 92.32% mean cross-validation accuracy and 92.05% mean cross-validation macro F1-score.

Although the results are promising for this dataset, the small sample size and absence of independent external validation mean that further testing is necessary before real-world use.
