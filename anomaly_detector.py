"""
Anomaly Detection Module
Flags individual features that are critically abnormal, even if overall prediction is healthy.
This acts as a safety net to catch dangerous outliers.
"""
import numpy as np

class AnomalyDetector:
    """
    Detects critical anomalies in individual blood test parameters.
    Uses z-score and clinical thresholds.
    """
    
    def __init__(self):
        # Critical thresholds (values that are medically dangerous)
        self.critical_high = {
            'Glucose': 180,  # Severe hyperglycemia
            'Cholesterol': 240,  # High risk
            'Hemoglobin': 18,  # Polycythemia risk
            'Platelets': 450000,  # Thrombocytosis
            'White Blood Cells': 11000,  # Leukocytosis
            'Hematocrit': 52,  # High risk
            'Insulin': 25,  # Hyperinsulinemia
            'BMI': 35,  # Obesity class 2
            'Systolic Blood Pressure': 180,  # Hypertensive crisis
            'Diastolic Blood Pressure': 120,  # Hypertensive crisis
            'Triglycerides': 200,  # High
            'HbA1c': 6.5,  # Diabetes threshold
            'LDL Cholesterol': 190,  # Very high
            'ALT': 100,  # Liver damage
            'AST': 100,  # Liver damage
            'Heart Rate': 120,  # Tachycardia
            'Creatinine': 1.5,  # Kidney dysfunction
            'Troponin': 0.04,  # Heart attack
            'C-reactive Protein': 10  # Severe inflammation
        }
        
        self.critical_low = {
            'Glucose': 60,  # Hypoglycemia
            'Hemoglobin': 10,  # Severe anemia
            'Platelets': 100000,  # Thrombocytopenia
            'White Blood Cells': 3000,  # Leukopenia
            'Red Blood Cells': 3.5,  # Anemia
            'Hematocrit': 30,  # Severe anemia
            'BMI': 16,  # Underweight
            'Systolic Blood Pressure': 90,  # Hypotension
            'Diastolic Blood Pressure': 60,  # Hypotension
            'HDL Cholesterol': 40,  # Low (increased risk)
            'Heart Rate': 50  # Bradycardia
        }
        
    def detect_anomalies(self, raw_features):
        """
        Detect critical anomalies in raw feature values.
        
        Args:
            raw_features: Dictionary of feature names to raw values
            
        Returns:
            List of anomaly dictionaries with:
                - feature: Feature name
                - value: Actual value
                - threshold: Critical threshold
                - severity: 'critical_high' or 'critical_low'
                - message: Human-readable warning
        """
        anomalies = []
        
        # Check critical high values
        for feature, threshold in self.critical_high.items():
            if feature in raw_features:
                value = raw_features[feature]
                if value >= threshold:
                    anomalies.append({
                        'feature': feature,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'critical_high',
                        'message': f'{feature} is critically HIGH ({value:.2f} ≥ {threshold})'
                    })
        
        # Check critical low values
        for feature, threshold in self.critical_low.items():
            if feature in raw_features:
                value = raw_features[feature]
                if value <= threshold:
                    anomalies.append({
                        'feature': feature,
                        'value': value,
                        'threshold': threshold,
                        'severity': 'critical_low',
                        'message': f'{feature} is critically LOW ({value:.2f} ≤ {threshold})'
                    })
        
        return anomalies
    
    def get_risk_level(self, anomalies):
        """
        Determine overall risk level based on anomalies.
        
        Returns:
            'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'
        """
        if not anomalies:
            return 'LOW'
        
        # Critical markers that require immediate attention
        critical_markers = ['Troponin', 'Glucose', 'Systolic Blood Pressure', 
                          'Diastolic Blood Pressure', 'Creatinine']
        
        for anomaly in anomalies:
            if anomaly['feature'] in critical_markers:
                return 'CRITICAL'
        
        if len(anomalies) >= 3:
            return 'HIGH'
        elif len(anomalies) >= 1:
            return 'MEDIUM'
        
        return 'LOW'


def test_anomaly_detector():
    """Test the anomaly detector"""
    detector = AnomalyDetector()
    
    # Test case 1: Normal values except critically high Troponin (heart attack)
    test_case_1 = {
        'Glucose': 95,
        'Cholesterol': 180,
        'Hemoglobin': 14.5,
        'Platelets': 250000,
        'White Blood Cells': 7000,
        'Red Blood Cells': 5.0,
        'Hematocrit': 42,
        'Troponin': 0.05,  # CRITICAL - indicates heart attack
        'Heart Rate': 72,
        'BMI': 22
    }
    
    print("Test Case 1: Normal except high Troponin (heart attack)")
    anomalies = detector.detect_anomalies(test_case_1)
    risk = detector.get_risk_level(anomalies)
    print(f"Risk Level: {risk}")
    for anomaly in anomalies:
        print(f"  ⚠️  {anomaly['message']}")
    
    # Test case 2: All normal
    test_case_2 = {
        'Glucose': 95,
        'Cholesterol': 180,
        'Hemoglobin': 14.5,
        'Troponin': 0.01,
        'BMI': 22
    }
    
    print("\nTest Case 2: All normal")
    anomalies = detector.detect_anomalies(test_case_2)
    risk = detector.get_risk_level(anomalies)
    print(f"Risk Level: {risk}")
    if not anomalies:
        print("  ✓ No anomalies detected")
    
    # Test case 3: Multiple anomalies
    test_case_3 = {
        'Glucose': 200,  # High
        'HbA1c': 8.0,  # Diabetic
        'BMI': 38,  # Obese
        'Systolic Blood Pressure': 185,  # Hypertensive crisis
        'Troponin': 0.01
    }
    
    print("\nTest Case 3: Multiple anomalies")
    anomalies = detector.detect_anomalies(test_case_3)
    risk = detector.get_risk_level(anomalies)
    print(f"Risk Level: {risk}")
    for anomaly in anomalies:
        print(f"  ⚠️  {anomaly['message']}")

if __name__ == "__main__":
    test_anomaly_detector()
