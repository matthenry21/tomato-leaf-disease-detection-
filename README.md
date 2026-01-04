🍅 Tomato Leaf Disease Detection using Deep Learning

A complete end-to-end deep learning project that detects tomato leaf diseases from images using a CNN model, trained on a reorganized subset of the PlantVillage dataset, and deploying the trained model as a web application using Streamlit.

📘 Notebooks Overview

1️⃣ Dataset Preparation

note book : 01_dataset_preparation.ipynb 

Detailed dataset explanation and processing logic are fully documented inside this notebook.

2️⃣ Model Training

note book : 02_model_training.ipynb
	•	Builds and trains a CNN model using TensorFlow / Keras
	•	Evaluates performance using:
	•	Accuracy & loss curves
	•	Confusion matrix
	•	Saves the trained model for deployment 

note : trained model 

🚀 Deployment (Streamlit App)

File : app.py

app.py loads the trained model and provides:
	•	Image upload interface
	•	Real-time disease prediction
	•	Confidence score
	•	Disease description and management tips

