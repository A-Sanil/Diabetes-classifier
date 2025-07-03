# 🩺 Diabetes Prediction using Machine Learning

![Diabetes Classifier Banner](https://images.unsplash.com/photo-1511174511562-5f97f4f4e0c8?auto=format&fit=crop&w=800&q=80)

## 📌 Project Overview
This project predicts whether a person is diabetic based on clinical and demographic features using multiple supervised machine learning models. The goal is to provide a robust, interpretable, and accurate prediction pipeline for diabetes risk assessment.

---

## 📊 Dataset
- **Source:** [Kaggle Diabetes Prediction Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- **Features:**
  - `age`: Age of the patient
  - `gender`: Male, Female, or Other
  - `hypertension`: 0 = No, 1 = Yes
  - `heart_disease`: 0 = No, 1 = Yes
  - `smoking_history`: Categorical (never, No Info, current, former, not current, ever)
  - `bmi`: Body Mass Index
  - `HbA1c_level`: Hemoglobin A1c level
  - `blood_glucose_level`: Blood glucose level
  - `diabetes`: 0 = No, 1 = Yes (target)

![Sample Data Table](https://raw.githubusercontent.com/nishitpatel01/Fake_News_Detection/main/images/sample_data.png)

---

## 🧠 Models Used
- **K-Nearest Neighbors (KNN)**
- **Decision Tree**
- **Extra Trees Classifier**
- **Logistic Regression**
- **Voting Ensemble (Super Majority)**

### 🔬 How scikit-learn Works
- **Preprocessing:**
  - Label encoding for categorical features (e.g., `smoking_history`, `gender`)
  - Feature scaling for models that require it (e.g., KNN, Logistic Regression)
- **Model Training:**
  - Each model is trained on the training set using `fit()`
  - Hyperparameter tuning is performed using `GridSearchCV` for KNN and Decision Tree
- **Prediction:**
  - Models predict on the test set using `predict()`
  - Ensemble combines predictions using a super majority rule
- **Evaluation:**
  - Accuracy, classification report, confusion matrix, and ROC curve are used for evaluation

![scikit-learn Pipeline](https://scikit-learn.org/stable/_static/ml_map.png)

---

## 🚀 How to Run the Project

### 1️⃣ Install Requirements
```bash
pip install -r requirements.txt
```

### 2️⃣ Train Models & Save Artifacts
Run the training script to train models and save `.pkl` files:
```bash
python Classifer.py
```

### 3️⃣ Web App (Optional)
To use the web interface, run:
```bash
python app.py
```
This will open a browser window for interactive predictions.

---

## 📈 Example Results

| Model                | Accuracy |
|----------------------|----------|
| KNN                  | 0.95     |
| Extra Trees          | 0.97     |
| Decision Tree        | 0.95     |
| Logistic Regression  | 0.96     |
| Voting Ensemble      | 0.97     |

![Confusion Matrix Example](https://scikit-learn.org/stable/_images/sphx_glr_plot_confusion_matrix_001.png)

---

## 🖼️ Example Prediction

**Input:**
- Age: 55
- Gender: Male
- Hypertension: 1
- Heart Disease: 0
- Smoking History: former
- BMI: 28.5
- HbA1c Level: 7.2
- Blood Glucose Level: 180

**Output:**
```
KNN Prediction: 1 (Diabetes)
Extra Trees Prediction: 1 (Diabetes)
Decision Tree Prediction: 1 (Diabetes)
Logistic Regression Prediction: 1 (Diabetes)
Voting Ensemble Prediction: 1 (Diabetes)
Super Majority Ensemble Prediction: 1 (Diabetes)
```

---

## 🛠️ Project Structure
```
Diabetes-classifier/
├── Classifer.py
├── app.py
├── predictor.py
├── requirements.txt
├── diabetes_prediction_dataset.csv
├── .gitignore
└── README.md
```

---

## 📚 References
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Kaggle Diabetes Dataset](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)
- [Ensemble Learning](https://scikit-learn.org/stable/modules/ensemble.html)

---

## 🧰 Tech Stack Used

- **Python 3.12**  
  Main programming language for all scripts and model development.
- **scikit-learn**  
  For machine learning models, preprocessing, hyperparameter tuning, and evaluation.
- **pandas & numpy**  
  For data manipulation and numerical operations.
- **matplotlib**  
  For plotting confusion matrices and ROC curves.
- **Flask**  
  For the optional web application interface.
- **joblib**  
  For saving and loading trained models and encoders.
- **Jupyter Notebook** (optional)  
  For exploratory data analysis and prototyping.

---

## 💡 Future Work
- Integrate SVM and XGBoost models
- Add a GUI using Streamlit
- Perform hyperparameter tuning for all models
- Deploy as an API with Flask or FastAPI
- Add model explainability with SHAP or LIME

---

