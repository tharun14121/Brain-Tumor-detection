![GitHub stars](https://img.shields.io/github/stars/tharun14121/Brain-Tumor-detection)
![GitHub forks](https://img.shields.io/github/forks/tharun14121/Brain-Tumor-detection)
![GitHub repo size](https://img.shields.io/github/repo-size/tharun14121/Brain-Tumor-detection)
![GitHub last commit](https://img.shields.io/github/last-commit/tharun14121/Brain-Tumor-detection)


# Brain Tumor Detection from MRI Images using Deep Learning

**Project Overview**

Brain tumor detection from MRI scans is a critical medical imaging task that can assist radiologists in early diagnosis and treatment planning. In this project, we design and evaluate a deep learning–based image classification system using Convolutional Neural Networks (CNNs) with transfer learning to automatically classify MRI brain images into tumor and non-tumor categories. The model leverages VGG16 pretrained on ImageNet, fine-tuned on a benchmark brain MRI dataset, achieving high accuracy and robust generalization.

**Objectives**
1. Automate brain tumor detection from MRI images
2. Reduce manual diagnostic effort
3. Build an interpretable and explainable CNN pipeline
4. Apply transfer learning for improved performance on limited medical data

**Dataset Information**

Dataset Name: Brain Tumor MRI Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

**Tech Stack & Tools**

Python | TensorFlow | Keras | CNN | VGG16 | Transfer Learning
NumPy | Pandas | Matplotlib | Seaborn | Scikit-learn

**Data Preprocessing**

1. Image resizing to 224 × 224
2. Pixel normalization (0–1 scaling)
3. Data augmentation:
   
   3.1 Rotation
   
   3.2 Zoom
   
   3.3 Horizontal flip
   
5. Train–validation split
6. Batch loading using ImageDataGenerator


**Model Architecture**

Base Model : VGG16

Why this model?

1. Proven performance in medical imaging
2. Strong low-level feature extraction
3. Efficient transfer learning capability

Followed by training, Evaluation Metrics and Results






