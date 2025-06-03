import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn_scaled, y_train_knn)
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

tree = DecisionTreeClassifier()
tree.fit(X_train_tree, y_train_tree)
y_pred_tree = tree.predict(X_test_tree)
print(f"Decision Tree Accuracy: {accuracy_score(y_test_tree, y_pred_tree):.4f}")
print(classification_report(y_test_tree, y_pred_tree))

# Voting Ensemble (KNN, Extra Trees, Decision Tree, Logistic Regression)
# Use scaled data for KNN and Logistic Regression, raw for others
voting_clf = VotingClassifier(estimators=[
    ('knn', KNeighborsClassifier(n_neighbors=5)),
    ('extra', ExtraTreesClassifier(n_estimators=100, random_state=42)),
    ('tree', DecisionTreeClassifier()),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
], voting='hard')

# For ensemble, use intersection of columns and scale where needed
# We'll use X_train for Extra Trees and Logistic Regression, and X_train_knn_scaled for KNN
# To keep things simple, use X_train (raw) for all except KNN and Logistic Regression, which use their own scaled versions

# Fit ensemble on unscaled X_train (for Extra Trees and Decision Tree), but KNN and Logistic Regression will internally scale
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

# --- User Input Prediction Section ---
print("\n--- Diabetes Prediction for Custom Input ---")
input_data = {}

# List the features used by your model (excluding 'diabetes', 'gender', 'smoking_history')
feature_names = [col for col in X.columns]
try:
    for feature in feature_names:
        val = input(f"Enter value for {feature}: ")
        try:
            val = float(val)
        except ValueError:
            print("Invalid input, please enter a number.")
            exit(1)
        input_data[feature] = [val]
except KeyboardInterrupt:
    print("\nInput cancelled by user.")
    exit(1)

# Get categorical inputs before predictions
smoking_history = input("Enter smoking_history (never, No Info, current, former, not current, ever): ")
try:
    smoking_encoded = le_smoke.transform([smoking_history])[0]
except Exception:
    print("Invalid smoking_history value. Using 0 (never) as default.")
    smoking_encoded = 0

gender = input("Enter gender (Male, Female, Other): ")
try:
    gender_encoded = le_gender_tree.transform([gender])[0]
except Exception:
    print("Invalid gender value. Using 0 (Female) as default.")
    gender_encoded = 0

# Prepare input for KNN (with smoking_history)
input_data_knn = input_data.copy()
input_data_knn['smoking_history'] = [smoking_encoded]
input_df_knn = pd.DataFrame(input_data_knn)
input_df_knn_scaled = scaler_knn.transform(input_df_knn)

# Prepare input for Extra Trees and Logistic Regression (no smoking_history column)
input_df = pd.DataFrame(input_data)
input_df_logreg = scaler_logreg.transform(input_df)

# Prepare input for Decision Tree (needs both gender and smoking_history)
input_data_tree = input_data.copy()
input_data_tree['smoking_history'] = [le_smoke_tree.transform([smoking_history])[0]]
input_data_tree['gender'] = [gender_encoded]
input_df_tree = pd.DataFrame(input_data_tree)

# Predict with all models
knn_pred = knn.predict(input_df_knn_scaled)[0]
print(f"KNN Prediction (1=Diabetes, 0=No Diabetes): {knn_pred}")

extra_pred = extra_trees.predict(input_df)[0]
print(f"Extra Trees Prediction (1=Diabetes, 0=No Diabetes): {extra_pred}")

tree_pred = tree.predict(input_df_tree)[0]
print(f"Decision Tree Prediction (1=Diabetes, 0=No Diabetes): {tree_pred}")

logreg_pred = logreg.predict(input_df_logreg)[0]
print(f"Logistic Regression Prediction (1=Diabetes, 0=No Diabetes): {logreg_pred}")

ensemble_pred = voting_clf.predict(input_df)[0]
print(f"Voting Ensemble Prediction (1=Diabetes, 0=No Diabetes): {ensemble_pred}")

# Super majority ensemble: require at least 3 out of 4 models to agree
votes = [knn_pred, extra_pred, tree_pred, logreg_pred]
num_ones = votes.count(1)
num_zeros = votes.count(0)

print("\n=== SUPER MAJORITY ENSEMBLE RESULT ===")
if num_ones >= 3:
    print("Super Majority Ensemble Prediction: 1 (Diabetes)")
elif num_zeros >= 3:
    print("Super Majority Ensemble Prediction: 0 (No Diabetes)")
else:
    print("Super Majority Ensemble Prediction: Could not classify")
print("======================================\n")
