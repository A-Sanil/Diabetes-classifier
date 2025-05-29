# Diabetes Prediction using Machine Learning

## 📌 Aim
The aim of this project is to predict whether a person is diabetic based on clinical and demographic features using various supervised machine learning models. The dataset is derived from real patient data including age, BMI, blood glucose, and other vital metrics.

---

## 🧾 Dataset Description

The dataset contains the following features:

| Feature             | Description                                  |
|---------------------|----------------------------------------------|
| `gender`            | Gender of the patient                        |
| `age`               | Age in years                                 |
| `hypertension`      | 0 = No, 1 = Yes                               |
| `heart_disease`     | 0 = No, 1 = Yes                               |
| `smoking_history`   | Categorical smoking status                   |
| `bmi`               | Body Mass Index                              |
| `HbA1c_level`       | 3-month average glucose level                |
| `blood_glucose_level` | Current glucose level                     |
| `diabetes`          | Target class (0 = No, 1 = Yes)               |

---

## 📂 Files in the Repository

| File Name                   | Description                                                |
|-----------------------------|------------------------------------------------------------|
| `diabetes_prediction_dataset.csv` | Dataset used for training and prediction             |
| `Classifer.py`              | Python script for training models and running predictions |
| `all_classifier_results.csv`| Output summary of classifier accuracies                   |
| `SVM_model.pkl`             | (Optional) Serialized SVM model (not yet used in script)  |
| `requirements.txt`          | List of required Python packages                          |
| `.gitattributes`            | Git line-ending and text normalization config             |

---

## 🧠 Machine Learning Models Used

The following classifiers were trained and evaluated using `scikit-learn`:

| Model                  | Accuracy   |
|------------------------|------------|
| Extra Trees Classifier | **0.9674** |
| K-Nearest Neighbors    | 0.95265    |
| Decision Tree Classifier | 0.9521  |
| Naive Bayes            | 0.90555    |

Additionally, a majority voting ensemble model combines the predictions of the above four classifiers for improved robustness.

---

## ⚙️ How to Run the Project

### 🧪 Step 1: Install Required Libraries

Run the following command in your terminal to install dependencies:

```bash
pip install -r requirements.txt




💡 Future Work
Integrate SVM model into predictions

Add a GUI using Streamlit

Perform hyperparameter tuning using GridSearchCV

Deploy as an API with Flask or FastAPI