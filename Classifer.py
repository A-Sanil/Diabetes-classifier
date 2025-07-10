"""
Diabetes Classifier - Machine Learning Pipeline
Description: Predicts diabetes using multiple ML models, with hyperparameter tuning, cross-validation, and ensemble methods.
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import joblib
try:
    from xgboost import XGBClassifier
    xgboost_available = True
except ImportError:
    xgboost_available = False

def load_data(input_file):
    """Load and preview the diabetes dataset."""
    df = pd.read_csv(input_file)
    print("Sample data:\n", df.head(3))
    return df

def prepare_encoders(df):
    """Fit label encoders for categorical features."""
    le_smoke = LabelEncoder().fit(df['smoking_history'])
    le_gender = LabelEncoder().fit(df['gender'])
    le_smoke_tree = LabelEncoder().fit(df['smoking_history'])
    le_gender_tree = LabelEncoder().fit(df['gender'])
    return le_smoke, le_gender, le_smoke_tree, le_gender_tree

def train_knn(X, y):
    """Train and tune KNN with scaling and cross-validation."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    param_grid = {'n_neighbors': [3, 5, 7, 9]}
    grid = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
    grid.fit(X_scaled, y)
    knn = grid.best_estimator_
    print(f"Best KNN n_neighbors: {grid.best_params_['n_neighbors']}")
    cv_scores = cross_val_score(knn, X_scaled, y, cv=5)
    print(f"KNN Cross-Validation Accuracy: {cv_scores.mean():.4f}")
    return knn, scaler

def train_extra_trees(X, y):
    """Train Extra Trees Classifier."""
    model = ExtraTreesClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_logreg(X, y):
    """Train and scale Logistic Regression."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    return model, scaler

def train_decision_tree(X, y):
    """Train and tune Decision Tree with cross-validation."""
    param_grid = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
    grid = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
    grid.fit(X, y)
    tree = grid.best_estimator_
    print(f"Best Decision Tree params: {grid.best_params_}")
    cv_scores = cross_val_score(tree, X, y, cv=5)
    print(f"Decision Tree Cross-Validation Accuracy: {cv_scores.mean():.4f}")
    return tree

def train_random_forest(X, y):
    """Train Random Forest Classifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def train_xgboost(X, y):
    """Train XGBoost Classifier if available."""
    if not xgboost_available:
        print("XGBoost is not installed. Skipping XGBoost model.")
        return None
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test, scaler=None, name="Model"):
    """Print accuracy and classification report."""
    if scaler:
        X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)
    print(f"{name} Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    return y_pred

def plot_roc_curve(model, X_test, y_test, scaler=None, name="Model"):
    """Plot ROC curve for a model."""
    if scaler:
        X_test = scaler.transform(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend()
    plt.show()

def main():
    # Load data
    input_file = 'diabetes_prediction_dataset.csv'
    df = load_data(input_file)
    y = df['diabetes']

    # Prepare encoders
    le_smoke, le_gender, le_smoke_tree, le_gender_tree = prepare_encoders(df)

    # KNN: drop gender, encode smoking_history
    X_knn = df.drop(['diabetes', 'gender'], axis=1).copy()
    X_knn['smoking_history'] = le_smoke.transform(df['smoking_history'])
    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
    knn, scaler_knn = train_knn(X_train_knn, y_train_knn)
    y_pred_knn = evaluate_model(knn, X_test_knn, y_test_knn, scaler_knn, name="KNN")

    # Extra Trees & Logistic Regression: drop gender and smoking_history
    X = df.drop(['diabetes', 'gender', 'smoking_history'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    extra_trees = train_extra_trees(X_train, y_train)
    y_pred_extra = evaluate_model(extra_trees, X_test, y_test, name="Extra Trees")
    logreg, scaler_logreg = train_logreg(X_train, y_train)
    y_pred_logreg = evaluate_model(logreg, X_test, y_test, scaler_logreg, name="Logistic Regression")

    # Decision Tree: encode both gender and smoking_history
    X_tree = df.drop(['diabetes'], axis=1).copy()
    X_tree['smoking_history'] = le_smoke_tree.transform(df['smoking_history'])
    X_tree['gender'] = le_gender_tree.transform(df['gender'])
    X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y, test_size=0.2, random_state=42)
    tree = train_decision_tree(X_train_tree, y_train_tree)
    y_pred_tree = evaluate_model(tree, X_test_tree, y_test_tree, name="Decision Tree")

    # Random Forest: drop gender and smoking_history
    rf = train_random_forest(X_train, y_train)
    y_pred_rf = evaluate_model(rf, X_test, y_test, name="Random Forest")

    # XGBoost: drop gender and smoking_history
    if xgboost_available:
        xgb = train_xgboost(X_train, y_train)
        y_pred_xgb = evaluate_model(xgb, X_test, y_test, name="XGBoost")
    else:
        xgb = None

    # Voting Ensemble (add Random Forest and XGBoost if available)
    ensemble_estimators = [
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('extra', ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ('tree', DecisionTreeClassifier()),
        ('logreg', LogisticRegression(max_iter=1000, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
    if xgboost_available:
        ensemble_estimators.append(('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)))
    voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='hard')
    voting_clf.fit(X_train, y_train)
    ensemble_pred = voting_clf.predict(X_test)
    print(f"Voting Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
    print(classification_report(y_test, ensemble_pred))

    # Confusion Matrix and ROC Curve for Logistic Regression
    cm = confusion_matrix(y_test, y_pred_logreg)
    print("Confusion Matrix (Logistic Regression):\n", cm)
    plot_roc_curve(logreg, X_test, y_test, scaler_logreg, name="Logistic Regression")

    # Load models and encoders from zip/rar file instead of saving new ones
    # Example: extract and load with joblib from the archive
    import os
    import rarfile
    archive_path = 'All pretrained model data.rar'
    if os.path.exists(archive_path):
        with rarfile.RarFile(archive_path) as rf:
            for name in rf.namelist():
                if name.endswith('.pkl'):
                    rf.extract(name, path='.')
        print('Extracted pretrained models from archive.')
    else:
        print('Archive with pretrained models not found.')

if __name__ == "__main__":
    main()
