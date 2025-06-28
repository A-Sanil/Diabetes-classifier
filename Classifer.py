import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import matplotlib.pyplot as plt
import joblib

# Load the dataset
input_file = 'diabetes_prediction_dataset.csv'
df = pd.read_csv(input_file)
print(df.head(3))

# Prepare encoders for categorical features
le_smoke = LabelEncoder()
le_gender = LabelEncoder()
le_smoke_tree = LabelEncoder()
le_gender_tree = LabelEncoder()

# Prepare data for KNN (drop gender, encode smoking_history)
X_knn = df.drop(['diabetes', 'gender'], axis=1).copy()
X_knn['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])
y = df['diabetes']
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)

# Scale for KNN
scaler_knn = StandardScaler()
X_train_knn_scaled = scaler_knn.fit_transform(X_train_knn)
X_test_knn_scaled = scaler_knn.transform(X_test_knn)

# --- Hyperparameter Tuning for KNN ---
knn_param_grid = {'n_neighbors': [3, 5, 7, 9]}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=5)
knn_grid.fit(X_train_knn_scaled, y_train_knn)
knn = knn_grid.best_estimator_
print(f"Best KNN n_neighbors: {knn_grid.best_params_['n_neighbors']}")

# Cross-validation for KNN
knn_cv_scores = cross_val_score(knn, X_train_knn_scaled, y_train_knn, cv=5)
print(f"KNN Cross-Validation Accuracy: {knn_cv_scores.mean():.4f}")

y_pred_knn = knn.predict(X_test_knn_scaled)
print(f"KNN Accuracy: {accuracy_score(y_test_knn, y_pred_knn):.4f}")
print(classification_report(y_test_knn, y_pred_knn))

# Prepare data for Extra Trees and Logistic Regression (drop gender and smoking_history)
X = df.drop(['diabetes', 'gender', 'smoking_history'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale for Logistic Regression
scaler_logreg = StandardScaler()
X_train_logreg = scaler_logreg.fit_transform(X_train)
X_test_logreg = scaler_logreg.transform(X_test)

extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees.fit(X_train, y_train)
y_pred_extra = extra_trees.predict(X_test)
print(f"Extra Trees Accuracy: {accuracy_score(y_test, y_pred_extra):.4f}")
print(classification_report(y_test, y_pred_extra))

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train_logreg, y_train)
y_pred_logreg = logreg.predict(X_test_logreg)
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_logreg):.4f}")
print(classification_report(y_test, y_pred_logreg))

# Prepare data for Decision Tree (encode both gender and smoking_history)
X_tree = df.drop(['diabetes'], axis=1).copy()
X_tree['smoking_history'] = le_smoke_tree.fit_transform(df['smoking_history'])
X_tree['gender'] = le_gender_tree.fit_transform(df['gender'])
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning for Decision Tree ---
tree_param_grid = {'max_depth': [3, 5, 7, None], 'min_samples_split': [2, 5, 10]}
tree_grid = GridSearchCV(DecisionTreeClassifier(), tree_param_grid, cv=5)
tree_grid.fit(X_train_tree, y_train_tree)
tree = tree_grid.best_estimator_
print(f"Best Decision Tree params: {tree_grid.best_params_}")

# Cross-validation for Decision Tree
tree_cv_scores = cross_val_score(tree, X_train_tree, y_train_tree, cv=5)
print(f"Decision Tree Cross-Validation Accuracy: {tree_cv_scores.mean():.4f}")

y_pred_tree = tree.predict(X_test_tree)
print(f"Decision Tree Accuracy: {accuracy_score(y_test_tree, y_pred_tree):.4f}")
print(classification_report(y_test_tree, y_pred_tree))

# Voting Ensemble (KNN, Extra Trees, Decision Tree, Logistic Regression)
voting_clf = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('extra', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('tree', DecisionTreeClassifier()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
], voting='hard')

voting_clf.fit(X_train, y_train)
ensemble_pred = voting_clf.predict(X_test)
print(f"Voting Ensemble Accuracy: {accuracy_score(y_test, ensemble_pred):.4f}")
print(classification_report(y_test, ensemble_pred))

# Confusion Matrix and ROC Curve for Logistic Regression
cm = confusion_matrix(y_test, y_pred_logreg)
print("Confusion Matrix (Logistic Regression):\n", cm)

y_score = logreg.predict_proba(X_test_logreg)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Save models and encoders
joblib.dump(knn, "knn.pkl")
joblib.dump(extra_trees, "extra_trees.pkl")
joblib.dump(tree, "tree.pkl")
joblib.dump(logreg, "logreg.pkl")
joblib.dump(voting_clf, "voting_clf.pkl")
joblib.dump(le_smoke, "le_smoke.pkl")
joblib.dump(le_smoke_tree, "le_smoke_tree.pkl")
joblib.dump(le_gender_tree, "le_gender_tree.pkl")
joblib.dump(scaler_knn, "scaler_knn.pkl")
