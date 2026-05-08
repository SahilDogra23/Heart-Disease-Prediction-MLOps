# 🫀 Heart Disease Prediction — Production ML System

An end-to-end Machine Learning system that predicts the presence of heart disease using clinical features. Built following the framework from **Aurélien Géron's "Hands-On Machine Learning" Chapter 2** and deployed to production.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B)
![Render](https://img.shields.io/badge/Render-Deployed-46E3B7)

## 🌐 Live Demo
| Component | URL |
|-----------|-----|
| 🖥️ Web App | [heart-disease-prediction.streamlit.app](https://heart-disease-prediction-bekd7nprm3xyizjtd2twp2.streamlit.app) |
| ⚙️ REST API | [heart-disease-api-jh78.onrender.com](https://heart-disease-api-jh78.onrender.com) |
| 📖 API Docs | [/docs](https://heart-disease-api-jh78.onrender.com/docs) |

## 🏗️ System Architecture
``` Raw Clinical Data (CSV)
↓
Data Cleaning & EDA
(Removed 723 duplicates)
↓
Model Training & Tuning
(GridSearchCV → 82% accuracy, 85% recall)
↓
Saved Model (.pkl)
↓
FastAPI REST API  ←→  Streamlit Web App
(Render)               (Streamlit Cloud) ```

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| Accuracy | 82% |
| Recall | 85% |
| Models Compared | Logistic Regression, Random Forest, SVM |
| Tuning Method | GridSearchCV (optimized for Recall) |

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML | Scikit-learn, Pandas, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| API Hosting | Render |
| Frontend Hosting | Streamlit Cloud |
| Version Control | Git + GitHub |

## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/SahilDogra23/Heart-Disease-Prediction-MLOps.git
cd Heart-Disease-Prediction-MLOps
pip install -r requirements.txt
```

### 2. Start the API
```bash
uvicorn app:app --reload
```

### 3. Start the frontend (new terminal)
```bash
streamlit run streamlit_app.py
```

### 4. Or call the API directly
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age":52,"sex":1,"cp":0,"trestbps":125,"chol":212,"fbs":0,"restecg":1,"thalach":168,"exang":0,"oldpeak":1.0,"slope":2,"ca":2,"thal":3}'
```
## 🐳 Docker Deployment

### Build the image
```bash
docker build -t heart-disease-api .
```

### Run the container
```bash
docker run -p 8000:8000 heart-disease-api
```

### Access the API
Visit `http://127.0.0.1:8000/docs`

## 📁 Project Structure
```Heart_Disease_prediction/
├── notebooks/
│   └── heart_disease.ipynb    ← EDA, training, evaluation
├── models/
│   ├── heart_disease_model.pkl
│   └── heart_disease_features.pkl
├── app.py                     ← FastAPI backend
├── streamlit_app.py           ← Streamlit frontend
├── Procfile                   ← Render deployment config
├── requirements.txt
└── README.md```

## 📚 Reference
- Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* — Chapter 2
- Dataset: [Kaggle Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset)

## 👤 Author
**Sahil Dogra**
[![GitHub](https://img.shields.io/badge/GitHub-SahilDogra23-black)](https://github.com/SahilDogra23)