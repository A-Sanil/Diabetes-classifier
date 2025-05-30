import pandas as pd
import sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.linear_model import LogisticRegression

input_file = 'diabetes_prediction_dataset.csv'
# Load the dataset
df = pd.read_csv(input_file)
# Display the first few rows of the dataset
print(df.head(3))

# Split the dataset into features and target variable
# Drop 'gender' and 'smoking_history' for all classifiers except Decision Tree (where they are encoded) and KNN (where gender is dropped and smoking_history is encoded)
X = df.drop(['diabetes', 'gender', 'smoking_history'], axis=1)
y = df['diabetes']
# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Initialize results dictionary at the top so all blocks can use it
results = {}
# Define the classifiers to be used

# --- Model training blocks below ---




# For KNN: drop 'gender', encode 'smoking_history'
X_knn = df.drop(['diabetes', 'gender'], axis=1).copy()
le_smoke = LabelEncoder()
X_knn['smoking_history'] = le_smoke.fit_transform(df['smoking_history'])
X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(X_knn, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_knn, y_train_knn)
y_pred_knn = knn.predict(X_test_knn)
accuracy_knn = accuracy_score(y_test_knn, y_pred_knn)
report_knn = classification_report(y_test_knn, y_pred_knn, output_dict=True)
results['KNN'] = {
    'accuracy': accuracy_knn,
    'report': report_knn
}
print(f"KNN (encoded smoking_history, gender dropped) Accuracy: {accuracy_knn:.4f}")
print(classification_report(y_test_knn, y_pred_knn))

# For Extra Trees: drop 'gender' and 'smoking_history' (already done in X)
extra_trees = ExtraTreesClassifier(n_estimators=100, random_state=42)
extra_trees.fit(X_train, y_train)
y_pred_extra = extra_trees.predict(X_test)
accuracy_extra = accuracy_score(y_test, y_pred_extra)
report_extra = classification_report(y_test, y_pred_extra, output_dict=True)
results['Extra Trees'] = {
    'accuracy': accuracy_extra,
    'report': report_extra
}
print(f"Extra Trees Accuracy: {accuracy_extra:.4f}")
print(classification_report(y_test, y_pred_extra))


# For Decision Tree: encode both 'smoking_history' and 'gender'
X_tree = df.drop(['diabetes'], axis=1).copy()
le_smoke_tree = LabelEncoder()
le_gender_tree = LabelEncoder()
X_tree['smoking_history'] = le_smoke_tree.fit_transform(df['smoking_history'])
X_tree['gender'] = le_gender_tree.fit_transform(df['gender'])
X_train_tree, X_test_tree, y_train_tree, y_test_tree = train_test_split(X_tree, y, test_size=0.2, random_state=42)
tree = DecisionTreeClassifier()
tree.fit(X_train_tree, y_train_tree)
y_pred_tree = tree.predict(X_test_tree)
accuracy_tree = accuracy_score(y_test_tree, y_pred_tree)
report_tree = classification_report(y_test_tree, y_pred_tree, output_dict=True)
results['Decision Tree'] = {
    'accuracy': accuracy_tree,
    'report': report_tree
}
print(f"Decision Tree (encoded smoking_history and gender) Accuracy: {accuracy_tree:.4f}")
print(classification_report(y_test_tree, y_pred_tree))



# For Naive Bayes: drop 'gender' and 'smoking_history' (already done in X)
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
accuracy_nb = accuracy_score(y_test, y_pred_nb)
report_nb = classification_report(y_test, y_pred_nb, output_dict=True)
results['Naive Bayes'] = {
    'accuracy': accuracy_nb,
    'report': report_nb
}
print(f"Naive Bayes Accuracy: {accuracy_nb:.4f}")
print(classification_report(y_test, y_pred_nb))

# For Logistic Regression: drop 'gender' and 'smoking_history' (already done in X)
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
accuracy_logreg = accuracy_score(y_test, y_pred_logreg)
report_logreg = classification_report(y_test, y_pred_logreg, output_dict=True)
results['Logistic Regression'] = {
    'accuracy': accuracy_logreg,
    'report': report_logreg
}
print(f"Logistic Regression Accuracy: {accuracy_logreg:.4f}")
print(classification_report(y_test, y_pred_logreg))

# Combine predictions: majority vote among Decision Tree, KNN, Extra Trees, and Logistic Regression
combined_pred = []
for dt, knn_pred, et, lr_pred in zip(y_pred_tree, y_pred_knn, y_pred_extra, y_pred_logreg):
    votes = [dt, knn_pred, et, lr_pred]
    if votes.count(1) > 2:  # majority of 4
        combined_pred.append(1)
    else:
        combined_pred.append(0)

print("Quad Model (Decision Tree, KNN, Extra Trees, Logistic Regression) Majority Vote Accuracy:", accuracy_score(y_test_tree, combined_pred))
print(classification_report(y_test_tree, combined_pred))

# Display all results in one line
print("\nAccuracies: KNN={:.4f}, DecisionTree={:.4f}, ExtraTrees={:.4f}, LogisticRegression={:.4f}, QuadModel={:.4f}".format(
    accuracy_knn, accuracy_tree, accuracy_extra, accuracy_logreg, accuracy_score(y_test_tree, combined_pred)))

# === Move results display and saving to the end ===
names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in names]
results_df = pd.DataFrame({
    'Classifier': names,
    'Accuracy': accuracies
})
print("\nClassifier Results:")

# Save results to a new file in the folder
results_df.to_csv('all_classifier_results.csv', index=False)
print("Results have also been saved to all_classifier_results.csv")
print(results_df)

# --- User Input Prediction Section ---
print("\n--- Diabetes Prediction for Custom Input ---")
input_data = {}

# List the features used by your model (excluding 'diabetes', 'gender', 'smoking_history')
feature_names = [col for col in X.columns]
for feature in feature_names:
    val = input(f"Enter value for {feature}: ")
    # Convert to float (or int) as needed
    try:
        val = float(val)
    except ValueError:
        pass
    input_data[feature] = [val]

# For KNN: need 'smoking_history' encoded, but Extra Trees and Logistic Regression do not use it
smoking_history = input("Enter smoking_history (never, No Info, current, former, not current, ever): ")
try:
    smoking_encoded = le_smoke.transform([smoking_history])[0]
except Exception:
    print("Invalid smoking_history value. Using 0 (never) as default.")
    smoking_encoded = 0

# Prepare input for KNN (with smoking_history)
input_data_knn = input_data.copy()
input_data_knn['smoking_history'] = [smoking_encoded]
input_df_knn = pd.DataFrame(input_data_knn)

# Prepare input for Extra Trees and Logistic Regression (no smoking_history column)
input_df = pd.DataFrame(input_data)

# Predict with KNN
knn_pred = knn.predict(input_df_knn)[0]
print(f"KNN Prediction (1=Diabetes, 0=No Diabetes): {knn_pred}")

# Predict with Extra Trees
extra_pred = extra_trees.predict(input_df)[0]
print(f"Extra Trees Prediction (1=Diabetes, 0=No Diabetes): {extra_pred}")

# Predict with Logistic Regression
logreg_pred = logreg.predict(input_df)[0]
print(f"Logistic Regression Prediction (1=Diabetes, 0=No Diabetes): {logreg_pred}")

# Visualization: Decision Boundary for Logistic Regression
# Select two features for visualization
feature_x = 'age'
feature_y = 'blood_glucose_level'

X_vis = df[[feature_x, feature_y]].values
y_vis = df['diabetes'].values

# Fit a classifier (e.g., Logistic Regression)
clf = LogisticRegression()
clf.fit(X_vis, y_vis)

# Create mesh grid
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                     np.arange(y_min, y_max, 0.5))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00'])
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_vis, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlabel(feature_x)
plt.ylabel(feature_y)
plt.title('Decision Boundary for Logistic Regression')
plt.show()

# === Move results display and saving to the end ===
names = list(results.keys())
accuracies = [results[name]['accuracy'] for name in names]
results_df = pd.DataFrame({
    'Classifier': names,
    'Accuracy': accuracies
})
print("\nClassifier Results:")
print(results_df)
