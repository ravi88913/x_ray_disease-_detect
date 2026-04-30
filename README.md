# x_ray_disease-_detect
# 🩺 Chest X-ray Disease Detection using Deep Learning

This project presents a deep learning-based system for detecting multiple diseases from chest X-ray images using **EfficientNetB0** and **transfer learning**. The system is capable of predicting multiple conditions from a single X-ray image and provides results through an interactive web interface.

---

## 🚀 Features

- Multi-label disease classification  
- Transfer learning using EfficientNetB0  
- Handles class imbalance using Focal Loss  
- Real-time prediction via Streamlit web app  
- Displays top predicted diseases with confidence scores  

---

## 🧠 Model Details

- **Base Model:** EfficientNetB0 (pretrained on ImageNet)  
- **Architecture:**
  - GlobalAveragePooling  
  - Dense (ReLU)  
  - Dropout  
  - Output layer (Sigmoid for multi-label classification)  

- **Loss Function:** Focal Loss  
- **Optimizer:** Adam  
- **Metrics:** AUC, Precision, Recall  

---

## 📊 Dataset

- NIH Chest X-ray Dataset (Kaggle)  
- Multi-label dataset with diseases such as:
  - Atelectasis  
  - Cardiomegaly  
  - Effusion  
  - Pneumonia  
  - Fibrosis  
  - No Finding  

---

## ⚙️ Preprocessing

- Images resized to **224 × 224**  
- Normalization applied  
- Multi-label encoding  
- Data augmentation:
  - Random Flip  
  - Rotation  
  - Zoom  

---

## 🏋️ Training Strategy

- Two-phase training:
  1. Freeze base model layers  
  2. Fine-tune last layers  

- Learning rate scheduling used  
- Threshold tuning (0.2 – 0.4) for better performance  

---

## 📈 Results

- **AUC:** ~0.83  
- High recall achieved (important for medical diagnosis)  
- Model performs well for multi-label predictions  

---

## 🌐 Web Application

A simple web interface is built using **Streamlit**:

### Features:
- Upload chest X-ray image  
- Automatic preprocessing  
- Real-time prediction  
- Displays top diseases with probabilities  

---

## 🖥️ How to Run

### 1. Install dependencies
```bash
pip install tensorflow streamlit numpy pillow
