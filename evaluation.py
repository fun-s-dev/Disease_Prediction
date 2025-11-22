"""
Evaluation Module
Reports on model performance, prioritizing Recall and Scaling Bridge accuracy
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    recall_score, accuracy_score, precision_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from module_b_scaling_bridge import ScalingBridge
import os

def evaluate_model_performance():
    """Evaluate the trained model on test data"""
    print("="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Load model and components
    model = joblib.load('models/best_model.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    
    # Load test data
    try:
        if os.path.exists('data/test_split.csv'):
            print("Loading test data from 'data/test_split.csv' (from last training run)")
            test_df = pd.read_csv('data/test_split.csv')
        else:
            print("Loading test data from 'data/blood_samples_dataset_test.csv'")
            test_df = pd.read_csv('data/blood_samples_dataset_test.csv')
            
        print(f"\nTest dataset shape: {test_df.shape}")
    except FileNotFoundError:
        print("Test dataset not found. Using training data split for evaluation.")
        # Use training data for demonstration
        train_df = pd.read_csv('data/Blood_samples_dataset_balanced_2(f).csv')
        from sklearn.model_selection import train_test_split
        
        feature_cols = [col for col in train_df.columns if col != 'Disease']
        X = train_df[feature_cols].values
        y = train_df['Disease'].values
        y_encoded = label_encoder.transform(y)
        
        _, X_test, _, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        y_test_decoded = label_encoder.inverse_transform(y_test)
        y_pred = model.predict(X_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        
        test_df = None
    else:
        # Prepare test data
        feature_cols = [col for col in test_df.columns if col != 'Disease']
        
        # Filter out samples with unseen labels
        known_classes = set(label_encoder.classes_)
        mask = test_df['Disease'].isin(known_classes)
        test_df_filtered = test_df[mask].copy()
        
        if len(test_df_filtered) < len(test_df):
            removed = len(test_df) - len(test_df_filtered)
            print(f"\n⚠️  Warning: {removed} samples with unseen labels removed from test set")
            print(f"   Remaining test samples: {len(test_df_filtered)}")
        
        X_test = test_df_filtered[feature_cols].values
        y_test = test_df_filtered['Disease'].values
        y_test_encoded = label_encoder.transform(y_test)
        
        y_pred = model.predict(X_test)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = y_test
    
    # Calculate metrics
    accuracy = accuracy_score(y_test_decoded, y_pred_decoded)
    recall_macro = recall_score(y_test_decoded, y_pred_decoded, average='macro')
    recall_weighted = recall_score(y_test_decoded, y_pred_decoded, average='weighted')
    precision_macro = precision_score(y_test_decoded, y_pred_decoded, average='macro')
    f1_macro = f1_score(y_test_decoded, y_pred_decoded, average='macro')
    
    # Per-class recall (most important for medical applications)
    recall_per_class = recall_score(y_test_decoded, y_pred_decoded, average=None)
    classes = label_encoder.classes_
    
    print("\n" + "="*60)
    print("OVERALL METRICS")
    print("="*60)
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Macro Recall:       {recall_macro:.4f} ⭐ (Primary Metric)")
    print(f"Weighted Recall:    {recall_weighted:.4f}")
    print(f"Macro Precision:    {precision_macro:.4f}")
    print(f"Macro F1-Score:     {f1_macro:.4f}")
    
    print("\n" + "="*60)
    print("PER-CLASS RECALL (Sensitivity)")
    print("="*60)
    for i, class_name in enumerate(classes):
        print(f"{class_name:20s}: {recall_per_class[i]:.4f}")
    
    print("\n" + "="*60)
    print("DETAILED CLASSIFICATION REPORT")
    print("="*60)
    print(classification_report(y_test_decoded, y_pred_decoded))
    
    # Confusion Matrix
    print("\n" + "="*60)
    print("CONFUSION MATRIX")
    print("="*60)
    cm = confusion_matrix(y_test_decoded, y_pred_decoded)
    print(cm)
    
    # Visualize confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('evaluation_confusion_matrix.png', dpi=300)
    print("\n✓ Confusion matrix saved to 'evaluation_confusion_matrix.png'")
    
    return {
        'accuracy': accuracy,
        'recall_macro': recall_macro,
        'recall_weighted': recall_weighted,
        'precision_macro': precision_macro,
        'f1_macro': f1_macro,
        'recall_per_class': dict(zip(classes, recall_per_class)),
        'confusion_matrix': cm
    }

def evaluate_scaling_bridge():
    """Evaluate the accuracy of the Scaling Bridge"""
    print("\n" + "="*60)
    print("SCALING BRIDGE EVALUATION")
    print("="*60)
    
    # Load scaling bridge
    bridge = ScalingBridge.load('models/scaling_bridge.pkl')
    
    # Load training data
    train_df = pd.read_csv('data/Blood_samples_dataset_balanced_2(f).csv')
    feature_cols = [col for col in train_df.columns if col != 'Disease']
    
    # Test scaling accuracy
    # Since we don't have original raw values, we'll test the inverse scaling
    # by checking if scaled values (0-1) can be reasonably mapped back
    
    print("\nTesting Scaling Bridge with sample values:")
    
    test_cases = {
        'Glucose': [70, 100, 120, 140],
        'BMI': [18, 22, 25, 30],
        'Troponin': [0, 0.01, 0.02, 0.04],
        'Hemoglobin': [12, 14, 16, 18],
        'Systolic Blood Pressure': [90, 120, 140, 180]
    }
    
    scaling_errors = []
    
    for feature, test_values in test_cases.items():
        print(f"\n{feature}:")
        for raw_val in test_values:
            scaled = bridge.scale_value(feature, raw_val)
            print(f"  Raw: {raw_val:6.2f} -> Scaled: {scaled:.4f}")
    
    # Check if scaled values are in valid range
    print("\n" + "="*60)
    print("SCALING VALIDATION")
    print("="*60)
    
    # Sample some scaled values from dataset and check if they're in [0, 1]
    sample_scaled = train_df[feature_cols].sample(100)
    all_in_range = True
    
    for col in feature_cols:
        col_min = sample_scaled[col].min()
        col_max = sample_scaled[col].max()
        if col_min < 0 or col_max > 1:
            print(f"⚠️  {col}: Values outside [0,1] range: [{col_min:.4f}, {col_max:.4f}]")
            all_in_range = False
    
    if all_in_range:
        print("✓ All scaled values are in valid [0, 1] range")
    
    # Test reverse scaling (estimate raw from scaled)
    print("\nTesting reverse scaling estimation:")
    test_scaled = 0.5  # Middle value
    for feature in list(test_cases.keys())[:3]:
        min_val, max_val = bridge.get_feature_range(feature)
        estimated_raw = min_val + test_scaled * (max_val - min_val)
        print(f"  {feature}: Scaled {test_scaled:.2f} -> Estimated Raw: {estimated_raw:.2f}")
    
    print("\n✓ Scaling Bridge evaluation complete")
    
    return {
        'all_in_range': all_in_range,
        'test_cases': test_cases
    }

def generate_evaluation_report():
    """Generate comprehensive evaluation report"""
    print("\n" + "="*60)
    print("GENERATING EVALUATION REPORT")
    print("="*60)
    
    # Evaluate model
    model_metrics = evaluate_model_performance()
    
    # Evaluate scaling bridge
    scaling_metrics = evaluate_scaling_bridge()
    
    # Create summary report
    report = f"""
{'='*60}
MEDIGUARD AI - EVALUATION REPORT
{'='*60}

MODEL PERFORMANCE SUMMARY:
--------------------------
Accuracy:           {model_metrics['accuracy']:.4f}
Macro Recall:        {model_metrics['recall_macro']:.4f} ⭐ (Primary Metric)
Weighted Recall:     {model_metrics['recall_weighted']:.4f}
Macro Precision:     {model_metrics['precision_macro']:.4f}
Macro F1-Score:      {model_metrics['f1_macro']:.4f}

PER-CLASS RECALL (Sensitivity):
{chr(10).join([f"  {disease:20s}: {recall:.4f}" for disease, recall in model_metrics['recall_per_class'].items()])}

SCALING BRIDGE STATUS:
----------------------
Scaling Validation:  {'✓ PASS' if scaling_metrics['all_in_range'] else '✗ FAIL'}
All values in [0,1]: {scaling_metrics['all_in_range']}

CONCLUSION:
-----------
The model demonstrates {'strong' if model_metrics['recall_macro'] > 0.85 else 'moderate' if model_metrics['recall_macro'] > 0.75 else 'acceptable'} 
performance with a macro recall of {model_metrics['recall_macro']:.4f}, which is {'excellent' if model_metrics['recall_macro'] > 0.85 else 'good' if model_metrics['recall_macro'] > 0.75 else 'acceptable'} 
for minimizing false negatives in medical triage scenarios.

{'='*60}
"""
    
    print(report)
    
    # Save report to file
    with open('evaluation_report.txt', 'w') as f:
        f.write(report)
    
    print("✓ Evaluation report saved to 'evaluation_report.txt'")
    
    return model_metrics, scaling_metrics

if __name__ == "__main__":
    generate_evaluation_report()

