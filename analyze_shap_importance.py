"""
SHAP Feature Importance Analysis
Shows which features dominate the model's predictions and provides recommendations.
"""
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from module_b_scaling_bridge import ScalingBridge

def analyze_shap_importance():
    """
    Analyze SHAP values to understand feature importance
    """
    print("="*70)
    print("SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load model components
    print("\nLoading model components...")
    shap_model = joblib.load('models/shap_model.pkl')  # XGBoost model for SHAP
    shap_explainer = joblib.load('models/shap_explainer.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    label_encoder = joblib.load('models/label_encoder.pkl')
    bridge = ScalingBridge.load('models/scaling_bridge.pkl')
    
    # Load test data
    test_df = pd.read_csv('data/test_split.csv')
    
    # Prepare features (add derived features)
    epsilon = 1e-6
    test_df['LDL_HDL_Ratio'] = test_df['LDL Cholesterol'] / (test_df['HDL Cholesterol'] + epsilon)
    test_df['Chol_HDL_Ratio'] = test_df['Cholesterol'] / (test_df['HDL Cholesterol'] + epsilon)
    test_df['Glucose_Insulin_Interaction'] = test_df['Glucose'] * test_df['Insulin']
    test_df['MAP'] = test_df['Diastolic Blood Pressure'] + (1/3 * (test_df['Systolic Blood Pressure'] - test_df['Diastolic Blood Pressure']))
    
    X_test = test_df[feature_names].values
    y_test = label_encoder.transform(test_df['Disease'].values)
    
    # Calculate SHAP values for test set (use subset for speed)
    print("Calculating SHAP values (this may take a moment)...")
    sample_size = min(100, len(X_test))
    X_sample = X_test[:sample_size]
    
    shap_values = shap_explainer(X_sample)
    
    # Get mean absolute SHAP values for each feature (across all classes)
    if len(shap_values.values.shape) == 3:
        # Multi-class: (samples, features, classes)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
    else:
        # Binary or single output
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    
    # Create feature importance DataFrame
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Abs_SHAP': mean_abs_shap
    }).sort_values('Mean_Abs_SHAP', ascending=False)
    
    print("\n" + "="*70)
    print("TOP 10 MOST IMPORTANT FEATURES (by mean absolute SHAP value)")
    print("="*70)
    print(f"\n{'Rank':<6} {'Feature':<35} {'Importance':<12}")
    print("-" * 70)
    
    for i, row in feature_importance.head(10).iterrows():
        print(f"{feature_importance.index.get_loc(i)+1:<6} {row['Feature']:<35} {row['Mean_Abs_SHAP']:<12.4f}")
    
    # Categorize features
    print("\n" + "="*70)
    print("FEATURE CATEGORIES")
    print("="*70)
    
    top_10_features = set(feature_importance.head(10)['Feature'].values)
    
    blood_markers = {'Hemoglobin', 'Platelets', 'White Blood Cells', 'Red Blood Cells', 
                     'Hematocrit', 'Mean Corpuscular Volume', 'Mean Corpuscular Hemoglobin',
                     'Mean Corpuscular Hemoglobin Concentration'}
    
    metabolic = {'Glucose', 'Insulin', 'HbA1c', 'Glucose_Insulin_Interaction'}
    
    cardiovascular = {'Systolic Blood Pressure', 'Diastolic Blood Pressure', 'MAP',
                     'Heart Rate', 'Troponin'}
    
    lipid = {'Cholesterol', 'LDL Cholesterol', 'HDL Cholesterol', 'Triglycerides',
             'LDL_HDL_Ratio', 'Chol_HDL_Ratio'}
    
    liver_kidney = {'ALT', 'AST', 'Creatinine'}
    
    other = {'BMI', 'C-reactive Protein'}
    
    categories = {
        'Blood Cell Markers': blood_markers & top_10_features,
        'Metabolic/Diabetes': metabolic & top_10_features,
        'Cardiovascular': cardiovascular & top_10_features,
        'Lipid Profile': lipid & top_10_features,
        'Liver/Kidney': liver_kidney & top_10_features,
        'Other': other & top_10_features
    }
    
    for category, features in categories.items():
        if features:
            print(f"\n{category}:")
            for feat in features:
                importance = feature_importance[feature_importance['Feature'] == feat]['Mean_Abs_SHAP'].values[0]
                print(f"  • {feat}: {importance:.4f}")
    
    # Recommendations
    print("\n" + "="*70)
    print("RECOMMENDATIONS")
    print("="*70)
    
    top_3 = feature_importance.head(3)['Feature'].values
    
    print("\n🎯 The model is MOST influenced by:")
    for i, feat in enumerate(top_3, 1):
        importance = feature_importance[feature_importance['Feature'] == feat]['Mean_Abs_SHAP'].values[0]
        print(f"   {i}. {feat} (importance: {importance:.4f})")
    
    print("\n💡 What to do about it:")
    print("\n1. DATA QUALITY:")
    print("   • Ensure these top features are measured ACCURATELY in clinical settings")
    print("   • Double-check data entry for these critical features")
    print("   • Consider additional validation for top 3 features")
    
    print("\n2. CLINICAL INTERPRETATION:")
    print("   • When explaining predictions to doctors, emphasize these features")
    print("   • Use SHAP API (/api/explain) to show feature contributions per patient")
    
    print("\n3. MODEL TRUST:")
    if any(feat in top_3 for feat in ['Hemoglobin', 'Glucose', 'HbA1c', 'Cholesterol']):
        print("   ✅ GOOD: Top features are clinically well-established disease markers")
        print("   → Model is learning medically sound patterns")
    
    if any(feat.endswith('_Ratio') or feat.endswith('_Interaction') for feat in top_3):
        print("   ✅ GOOD: Engineered features (ratios) are important")
        print("   → Feature engineering was successful")
    
    print("\n4. POTENTIAL CONCERNS:")
    # Check if derived features dominate too much
    derived_count = sum(1 for f in top_10_features if '_' in f and f not in feature_names[:24])
    if derived_count > 5:
        print("   ⚠️  Many derived features in top 10")
        print("   → Consider if model is over-relying on engineered features")
    else:
        print("   ✅ Balanced mix of raw and engineered features")
    
    print("\n5. ACTIONABLE STEPS:")
    print("   • Focus quality control on top 5 features")
    print("   • Train medical staff on importance of accurate measurement")
    print("   • Use SHAP explanations in production to build trust")
    print("   • Monitor prediction confidence when top features are missing/abnormal")
    
    # Save summary
    feature_importance.to_csv('shap_feature_importance.csv', index=False)
    print(f"\n✓ Full feature importance saved to 'shap_feature_importance.csv'")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    
    return feature_importance

if __name__ == "__main__":
    analyze_shap_importance()
