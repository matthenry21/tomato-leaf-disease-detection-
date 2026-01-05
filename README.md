# ☘️ Tomato Leaf Disease Detection using Deep Learning
A complete end-to-end deep learning project that detects tomato leaf diseases from images using a CNN model, trained on a reorganized subset of the PlantVillage dataset, and deployed the trained model as a web application using Streamlit.

## Problem Statement:
Tomato is one of the most widely cultivated and economically important crops in agriculture.However, tomato plants are highly susceptible to various leaf diseases such as Early Blight, Late Blight, Septoria Leaf Spot, Leaf Mold, and Bacterial Spot.These diseases significantly affect plant health, leading to reduced crop yield, poor fruit quality, and economic losses for farmers. Farmers often find it difficult to identify these diseases at an early stage, and expert help is not always available. Because of late detection, diseases spread quickly and cause major crop loss. Sometimes farmers also use the wrong or excess pesticides, which increases cost and harms the environment.

To solve this problem, I built a Convolutional Neural Network (CNN) model to detect tomato leaf diseases from leaf images. The CNN automatically learns important features from the images and accurately classifies different types of leaf diseases.I deployed the trained CNN model using a Streamlit web application. This system helps farmers and agricultural workers detect diseases early and take timely action, reducing crop loss and improving productivity.



---

## 📘 Notebooks Overview

#### **1️⃣ Dataset Preparation**
 
 note book : 01_dataset_preparation.ipynb
 
Detailed dataset explanation and processing logic are fully documented inside this notebook.

#### **2️⃣ Model Training**

Notebook: 02_model_training.ipynb

Key steps:
	
	•	Image loading using ImageDataGenerator
	•	CNN architecture with:
	•	Convolution layers
	•	ReLU activation
	•	MaxPooling
	•	Flatten
	•	Dense layers
	•	Training with:
	•	Categorical Crossentropy
	•	Adam optimizer
	•	Evaluation using:
	•	Accuracy & Loss curves
	•	Confusion Matrix

📌 The trained model is saved as:  model/tomato_model.h5 

**note** : trained model  was not added in repository due to its large size and was added in **releases**👈
please refer in my ***releases*** to understand how to load trained model for testing my deployed streamlite application.

---

## 🚀 Deployment (Streamlit App)

**File : app.py**

app.py loads the trained model from model/ directory and provides:
    
	•	Image upload interface
	•	Real-time disease prediction
	•	Confidence score
	•	Disease description and management tips

##  deployed streamlit webapp prediction Pipeline
	1.	User uploads an tomato leaf image(.png etc)
	2.	Image is resized to training input size
	3.	Image is normalized
	4.	Model predicts disease class
	5.	Output includes:
	•	Disease name
	•	Confidence score
	•	Disease information


---
## 🛠️ Tech Stack
    •	Python
	•	TensorFlow / Keras
	•	NumPy
	•	Matplotlib
	•	Streamlit
	•	Jupyter Notebook
---
## Conclusion

This project demonstrates:
	
	•	Real-world dataset handling
	•	Proper ML workflow separation
	•	CNN-based image classification
	•	End-to-end deployment

It is designed to be scalable, reproducible, and production-ready.
