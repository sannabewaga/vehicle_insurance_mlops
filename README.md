# Vehicle Insurance Prediction (End-to-End MLOps Project)

This project predicts whether a customer will purchase vehicle insurance based on demographic and vehicle-related features. It demonstrates an end-to-end machine learning workflow integrated with MLOps best practices such as CI/CD, Dockerization, and model tracking.

---

## 🔧 Tech Stack

- **FastAPI** – for building the REST API  
- **Scikit-learn** – for model training and evaluation  
- **MLflow** – for experiment tracking  
- **Docker** – for containerizing the application  
- **GitHub Actions** – for CI/CD automation  
- **MongoDB** – for storing input/output data  
- **AWS EC2** – cloud hosting and deployment (Docker-based)  

---

## 🚀 Features

- Cleaned and preprocessed vehicle insurance dataset  
- Trained multiple ML models with scikit-learn  
- Evaluated models using precision, recall, and F1-score  
- Tracked experiments and metrics using MLflow  
- Built REST API using FastAPI for model inference  
- Dockerized the application for portability  
- Integrated CI/CD with GitHub Actions  
- Connected API with MongoDB for input logging  

---

## 📊 Model Performance (Example)

**Class 0:**  
- Precision: 0.93  
- Recall: 0.75  
- F1-score: 0.83  

**Class 1:**  
- Precision: 0.85  
- Recall: 0.96  
- F1-score: 0.91  

**Overall Accuracy:** ~89%
