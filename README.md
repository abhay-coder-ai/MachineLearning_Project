# 🎓 Math Score Predictor – End-to-End Machine Learning Project

This project is a complete **end-to-end Machine Learning application** that predicts a student’s **math score** based on demographic and academic-related features.  
It covers the **full ML lifecycle** — from data preprocessing and model training to **production deployment**.

🔗 **Live Application:**  
https://machinelearning-project-j8rk.onrender.com

---

## 📌 Project Overview

The goal of this project is to build a real-world ML system that:
- Takes user input through a web interface
- Applies the same preprocessing used during training
- Uses a trained ML model to make predictions
- Runs reliably in a production environment

Unlike notebook-only projects, this one focuses heavily on **deployment stability, debugging, and MLOps fundamentals**.

---

## 🧠 Features Used for Prediction

- Gender  
- Race/Ethnicity  
- Parental level of education  
- Lunch type  
- Test preparation course  
- Reading score  
- Writing score  

---


## ⚙️ Tech Stack

- **Python**
- **Scikit-learn**
- **CatBoost**
- **Flask**
- **Docker**
- **Render (Cloud Deployment)**
- **Git & GitHub**

---

## 🚀 Model & Pipeline

- Modular pipeline design:
  - Data ingestion
  - Data transformation
  - Model training
- Preprocessing handled using:
  - `SimpleImputer`
  - `StandardScaler`
  - `OneHotEncoder`
- Model trained and saved as artifacts
- Same artifacts reused during inference to ensure consistency

---

## 🧩 Key Challenges & Learnings

This project involved solving **real production-level problems**, including:

- Silent training failures due to logging misconfiguration
- Corrupted model and preprocessor artifacts
- scikit-learn version incompatibility across environments
- Python runtime mismatch on cloud (Python 3.13 vs ML libraries)
- Missing and misleading deployment logs
- Automatic deployments causing instability

### 🔑 Final Solution
To make the system stable:
- Logging was fixed to prevent silent failures
- Artifacts were regenerated and validated
- Dependencies were strictly pinned
- The application was **containerized using Docker**
- Deployment was done using Docker on Render with Python 3.10

---

## 🐳 Docker Deployment (Why Docker?)

Docker was used to:
- Fully control the Python version
- Avoid dependency incompatibilities
- Ensure reproducible builds
- Make the deployment production-safe

This reflects **industry-standard ML deployment practices**.

