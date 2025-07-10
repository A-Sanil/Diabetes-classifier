import os
import pandas as pd
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'All pretrained model data')

# Load encoders and scalers
le_smoke = joblib.load(os.path.join(MODEL_DIR, "le_smoke.pkl"))
le_gender_tree = joblib.load(os.path.join(MODEL_DIR, "le_gender_tree.pkl"))
le_smoke_tree = joblib.load(os.path.join(MODEL_DIR, "le_smoke_tree.pkl"))
scaler_knn = joblib.load(os.path.join(MODEL_DIR, "scaler_knn.pkl"))
scaler_logreg = joblib.load(os.path.join(MODEL_DIR, "scaler_logreg.pkl"))

# Load models
knn = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
extra_trees = joblib.load(os.path.join(MODEL_DIR, "extra_trees.pkl"))
tree = joblib.load(os.path.join(MODEL_DIR, "tree.pkl"))
logreg = joblib.load(os.path.join(MODEL_DIR, "logreg.pkl"))
voting_clf = joblib.load(os.path.join(MODEL_DIR, "voting_clf.pkl"))
random_forest = joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl"))
xgboost = joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl"))

# List features (excluding 'diabetes', 'gender', 'smoking_history')
feature_names = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level', 'blood_glucose_level']

def predict_all(input_data, smoking_history, gender):
    # Prepare input for KNN
    input_data_knn = input_data.copy()
    input_data_knn['smoking_history'] = [le_smoke.transform([smoking_history])[0]]
    input_df_knn = pd.DataFrame(input_data_knn)
    input_df_knn_scaled = scaler_knn.transform(input_df_knn)

    # Prepare input for Extra Trees, Logistic Regression, Random Forest, XGBoost
    input_df = pd.DataFrame(input_data)
    input_df_logreg = scaler_logreg.transform(input_df)

    # Prepare input for Decision Tree
    input_data_tree = input_data.copy()
    input_data_tree['smoking_history'] = [le_smoke_tree.transform([smoking_history])[0]]
    if gender.lower() == "male":
        gender_encoded = 0
    elif gender.lower() == "female":
        gender_encoded = 1
    elif gender.lower() == "other":
        gender_encoded = 2
    else:
        gender_encoded = 0  # default to Male
    input_data_tree['gender'] = [gender_encoded]
    input_df_tree = pd.DataFrame(input_data_tree)

    # Predict
    knn_pred = knn.predict(input_df_knn_scaled)[0]
    extra_pred = extra_trees.predict(input_df)[0]
    tree_pred = tree.predict(input_df_tree)[0]
    logreg_pred = logreg.predict(input_df_logreg)[0]
    rf_pred = random_forest.predict(input_df)[0]
    xgb_pred = xgboost.predict(input_df)[0]
    ensemble_pred = voting_clf.predict(input_df)[0]

    # Super majority
    votes = [knn_pred, extra_pred, tree_pred, logreg_pred, rf_pred, xgb_pred]
    num_ones = votes.count(1)
    num_zeros = votes.count(0)
    if num_ones >= 4:
        super_majority = "1 (Diabetes)"
    elif num_zeros >= 4:
        super_majority = "0 (No Diabetes)"
    else:
        super_majority = "Could not classify"

    return {
        "KNN": knn_pred,
        "Extra Trees": extra_pred,
        "Decision Tree": tree_pred,
        "Logistic Regression": logreg_pred,
        "Random Forest": rf_pred,
        "XGBoost": xgb_pred,
        "Voting Ensemble": ensemble_pred,
        "Super Majority": super_majority
    }