"""
Module A: Train Multi-Class Classification Model
Trains XGBoost and Random Forest models with focus on high Recall (Sensitivity)
to minimize dangerous False Negatives.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, accuracy_score
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib
import os

def load_data(data_path):
    """Load the training dataset"""
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Disease distribution:\n{df['Disease'].value_counts()}")
    return df

def prepare_data(df):
    """Prepare features and target"""
    # Feature Engineering: Clinical Ratios
    # Add small epsilon to avoid division by zero
    epsilon = 1e-6
    
    df['LDL_HDL_Ratio'] = df['LDL Cholesterol'] / (df['HDL Cholesterol'] + epsilon)
    df['Chol_HDL_Ratio'] = df['Cholesterol'] / (df['HDL Cholesterol'] + epsilon)
    df['Glucose_Insulin_Interaction'] = df['Glucose'] * df['Insulin']
    
    # Mean Arterial Pressure (MAP) approx
    # MAP = DP + 1/3(SP - DP)
    df['MAP'] = df['Diastolic Blood Pressure'] + (1/3 * (df['Systolic Blood Pressure'] - df['Diastolic Blood Pressure']))
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col != 'Disease']
    X = df[feature_cols].values
    y = df['Disease'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, feature_cols, label_encoder

def train_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """Train XGBoost model with Hyperparameter Tuning"""
    print("\n" + "="*50)
    print("Training XGBoost Model (with Tuning)")
    print("="*50)
    
    # Define parameter distribution
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'scale_pos_weight': [1, 2, 5] # Help with imbalance
    }
    
    xgb_clf = xgb.XGBClassifier(
        random_state=42,
        eval_metric='mlogloss'
    )
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        estimator=xgb_clf,
        param_distributions=param_dist,
        n_iter=10, # Limit iterations for speed
        scoring='recall_weighted',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("Running RandomizedSearchCV...")
    random_search.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    print(f"Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

def train_random_forest(X_train, y_train, feature_names):
    """Train Random Forest model with Hyperparameter Tuning"""
    print("\n" + "="*50)
    print("Training Random Forest Model (with Tuning)")
    print("="*50)
    
    # Define parameter distribution
    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample', None]
    }
    
    rf_clf = RandomForestClassifier(random_state=42)
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        estimator=rf_clf,
        param_distributions=param_dist,
        n_iter=10, # Limit iterations for speed
        scoring='recall_weighted',
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )
    
    print("Running RandomizedSearchCV...")
    random_search.fit(X_train, y_train)
    
    print(f"Best parameters: {random_search.best_params_}")
    return random_search.best_estimator_

def evaluate_model(model, X_test, y_test, label_encoder, model_name):
    """Evaluate model performance with focus on Recall"""
    y_pred = model.predict(X_test)
    
    # Decode predictions
    y_test_decoded = label_encoder.inverse_transform(y_test)
    y_pred_decoded = label_encoder.inverse_transform(y_pred)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='weighted')
    recall_macro = recall_score(y_test, y_pred, average='macro')
    
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Macro Recall: {recall_macro:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test_decoded, y_pred_decoded))
    
    return {
        'accuracy': accuracy,
        'recall_weighted': recall,
        'recall_macro': recall_macro,
        'y_test': y_test_decoded,
        'y_pred': y_pred_decoded
    }

def main():
    """Main training pipeline"""
    # Load data
    # Load and combine datasets
    train_path = 'data/Blood_samples_dataset_balanced_2(f).csv'
    test_path = 'data/blood_samples_dataset_test.csv'
    
    print(f"Loading training data from {train_path}")
    df_train = load_data(train_path)
    
    if os.path.exists(test_path):
        print(f"Loading test data from {test_path}")
        df_test = pd.read_csv(test_path)
        print(f"Test data shape: {df_test.shape}")
        
        # Concatenate
        df = pd.concat([df_train, df_test], ignore_index=True)
        print(f"Combined dataset shape: {df.shape}")
    else:
        print("Warning: Test dataset not found. Using only training data.")
        df = df_train
    
    # Deduplicate data
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"\nRemoved {initial_rows - len(df)} duplicate rows")
    print(f"Unique samples for training: {len(df)}")

    # --- Data Augmentation for Healthy Class ---
    print("\nAugmenting data with synthetic 'Healthy' samples...")
    from module_b_scaling_bridge import ScalingBridge
    
    # Initialize bridge (using training data to estimate ranges if needed, but we use physiological)
    bridge = ScalingBridge(train_path) 
    
    n_synthetic = 1000
    synthetic_data = []
    
    # Physiological ranges from ScalingBridge
    ranges = bridge.physiological_ranges
    
    np.random.seed(42)
    
    for _ in range(n_synthetic):
        sample = {}
        for feature, (min_val, max_val) in ranges.items():
            # Generate random value within ideal range
            # Use a normal distribution centered in the range for more realism, 
            # or uniform for broader coverage. Uniform is safer to force the whole range.
            raw_val = np.random.uniform(min_val, max_val)
            sample[feature] = raw_val
            
        # Scale the sample
        scaled_sample = bridge.scale_features(sample)
        scaled_sample['Disease'] = 'Healthy'
        synthetic_data.append(scaled_sample)
        
    synthetic_df = pd.DataFrame(synthetic_data)
    
    # Ensure columns match
    synthetic_df = synthetic_df[df.columns]
    
    # Combine
    df = pd.concat([df, synthetic_df], ignore_index=True)
    print(f"Added {n_synthetic} synthetic Healthy samples")
    print(f"New dataset shape: {df.shape}")
    # -------------------------------------------

    # Prepare data
    X, y, feature_names, label_encoder = prepare_data(df)
    
    # Split data: 80% Train, 20% Test
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Split Train into Train and Validation (80% Train, 20% Val of the training set)
    # We keep Val pure (no SMOTE) for honest early stopping/evaluation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    
    print(f"\nTrain set (Before SMOTE): {X_train.shape[0]} samples")
    
    # Apply SMOTE to Training set ONLY
    # Use aggressive sampling for minority classes to improve recall
    print("Applying SMOTE to balance training classes...")
    
    # Calculate class distribution
    unique, counts = np.unique(y_train, return_counts=True)
    class_dist = dict(zip(unique, counts))
    max_samples = max(counts)
    
    # Create custom sampling strategy: oversample ALL minority classes to match majority
    sampling_strategy = {}
    for class_idx, count in class_dist.items():
        if count < max_samples:
            # Oversample to 80% of majority class (balanced approach)
            sampling_strategy[class_idx] = int(max_samples * 0.8)
    
    print(f"SMOTE sampling strategy: {sampling_strategy}")
    
    smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy, k_neighbors=3)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Train set (After SMOTE):  {X_train_resampled.shape[0]} samples")
    print(f"Validation set:           {X_val.shape[0]} samples")
    print(f"Test set:                 {X_test.shape[0]} samples")
    
    # Save test split for evaluation
    print("\nSaving test split to 'data/test_split.csv'...")
    test_df_save = pd.DataFrame(X_test, columns=feature_names)
    test_df_save['Disease'] = label_encoder.inverse_transform(y_test)
    test_df_save.to_csv('data/test_split.csv', index=False)
    
    # Train models using Resampled Training data
    xgb_model = train_xgboost(X_train_resampled, y_train_resampled, X_val, y_val, feature_names)
    rf_model = train_random_forest(X_train_resampled, y_train_resampled, feature_names)
    
    # Evaluate models
    xgb_results = evaluate_model(xgb_model, X_test, y_test, label_encoder, "XGBoost")
    rf_results = evaluate_model(rf_model, X_test, y_test, label_encoder, "Random Forest")
    
    # --- Ensemble Learning (Voting Classifier) ---
    print("\n" + "="*50)
    print("Training Voting Classifier (Ensemble)")
    print("="*50)
    
    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb_model), ('rf', rf_model)],
        voting='soft',
        weights=[2, 1]  # Give XGBoost more weight (better at minority classes)
    )
    
    voting_clf.fit(X_train_resampled, y_train_resampled)
    voting_results = evaluate_model(voting_clf, X_test, y_test, label_encoder, "Voting Ensemble")
    
    # Select best model (Voting is usually best, but let's be safe)
    # Actually, for this task, we prioritize the Ensemble for robustness
    best_model = voting_clf
    print(f"\n✓ Selected Voting Ensemble as best model (Recall: {voting_results['recall_weighted']:.4f})")
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    # --- SHAP Explainability ---
    print("\nGenerating SHAP Explainer...")
    # SHAP works best with the underlying XGBoost model
    # We save the XGBoost component specifically for explanations
    joblib.dump(xgb_model, 'models/shap_model.pkl')
    
    try:
        # Use TreeExplainer for XGBoost
        explainer = shap.TreeExplainer(xgb_model)
        joblib.dump(explainer, 'models/shap_explainer.pkl')
        print("✓ SHAP explainer saved to 'models/shap_explainer.pkl'")
    except Exception as e:
        print(f"Warning: Could not save SHAP explainer: {e}")
    
    print(f"\n✓ Model saved to 'models/best_model.pkl'")
    print(f"✓ Label encoder saved to 'models/label_encoder.pkl'")
    print(f"✓ Feature names saved to 'models/feature_names.pkl'")
    
    return best_model, label_encoder, feature_names, xgb_results, rf_results

if __name__ == "__main__":
    main()

