Intern ID : CITS8718


# 🌸 Iris Intelligence — Iris Species Classification

An interactive machine learning web application that predicts Iris flower
species from sepal and petal measurements.

The project combines machine learning, data exploration, model evaluation,
and interactive visualization into a single Streamlit dashboard.

---

## 🚀 Live Demo

**Streamlit Dashboard:**  
PASTE-YOUR-STREAMLIT-URL-HERE

---

## 📌 Project Overview

Iris Intelligence uses a **Random Forest Classifier** to classify Iris flowers
into three species:

- Iris Setosa
- Iris Versicolor
- Iris Virginica

Users can enter flower measurements and receive a predicted species together
with the model's confidence.

---

## ✨ Features

### 🔮 Interactive Prediction
Enter:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The application returns:

- Predicted species
- Prediction confidence
- Species probability distribution
- Measurement profile

### 📊 Data Analytics

Explore:

- Species distribution
- Feature relationships
- Measurement distributions
- Box plots
- Correlation matrix
- Species-level statistical summary

### 🤖 Model Intelligence

The dashboard provides:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- Classification Report
- Feature Importance

---

## 🧠 Machine Learning Workflow

```text
Iris Dataset
     ↓
Data Preparation
     ↓
Train / Test Split
     ↓
Random Forest Classifier
     ↓
Model Evaluation
     ↓
Species Prediction
     ↓
Interactive Streamlit Dashboard




🛠️ Technologies Used
Python
Pandas
NumPy
Scikit-learn
Plotly
Streamlit


📂 Project Structure
IRIS species classification/
│
├── app.py
├── iris.csv
├── requirements.txt
├── README.md
└── .gitignore



▶️ Run Locally

Clone the repository:
git clone YOUR-GITHUB-REPOSITORY-URL


Move into the project directory:
cd IRIS-species-classification


Install dependencies:
pip install -r requirements.txt

Run the application:
streamlit run app.py


📈 Model

The application uses a Random Forest Classifier trained on four numerical
features:
Sepal Length
Sepal Width
Petal Length
Petal Width
The dataset contains 150 observations across 3 Iris species.



🎯 Objective

The objective of this project is to demonstrate an end-to-end machine learning
workflow:

Data → Training → Prediction → Evaluation → Visualization → Deployment



👨‍💻 Author

Talha Ali

Built as a machine learning portfolio project.