"""
Module B: Scaling Bridge
Converts raw clinical values to 0-1 scaled format required by the ML model.
Approximates min/max values from the training dataset.
"""

import pandas as pd
import numpy as np
import joblib

class ScalingBridge:
    """
    Scaling Bridge that maps raw clinical values to 0-1 scaled format.
    Uses inverse scaling based on approximated min/max from training data.
    """
    
    def __init__(self, data_path=None):
        """
        Initialize Scaling Bridge
        
        Args:
            data_path: Path to training dataset to estimate min/max values
        """
        # Standard physiological ranges for blood test parameters
        # These are approximate ranges in common units
        self.physiological_ranges = {
            'Glucose': (70, 140),  # mg/dL
            'Cholesterol': (125, 200),  # mg/dL
            'Hemoglobin': (12, 18),  # g/dL
            'Platelets': (150000, 450000),  # per microliter of blood
            'White Blood Cells': (4000, 11000),  # per cubic millimeter of blood
            'Red Blood Cells': (4.0, 6.1),  # million cells per microliter of blood
            'Hematocrit': (36, 54),  # percentage
            'Mean Corpuscular Volume': (80, 100),  # femtoliters
            'Mean Corpuscular Hemoglobin': (27, 33),  # picograms
            'Mean Corpuscular Hemoglobin Concentration': (32, 36),  # grams per deciliter
            'Insulin': (2, 25),  # microU/mL
            'BMI': (18.5, 30),  # kg/m^2
            'Systolic Blood Pressure': (90, 140),  # mmHg
            'Diastolic Blood Pressure': (60, 90),  # mmHg
            'Triglycerides': (50, 200),  # mg/dL
            'HbA1c': (4, 6.5),  # percentage
            'LDL Cholesterol': (50, 130),  # mg/dL
            'HDL Cholesterol': (40, 80),  # mg/dL
            'ALT': (7, 56),  # U/L
            'AST': (10, 40),  # U/L
            'Heart Rate': (60, 100),  # beats per minute
            'Creatinine': (0.6, 1.3),  # mg/dL
            'Troponin': (0, 0.04),  # ng/mL
            'C-reactive Protein': (0, 10),  # mg/L
            # Derived Features
            'LDL_HDL_Ratio': (0.5, 7.0),
            'Chol_HDL_Ratio': (1.0, 10.0),
            'Glucose_Insulin_Interaction': (350, 3500),
            'MAP': (60, 100)  # Mean Arterial Pressure
        }
        
        # Estimate min/max from dataset if provided
        if data_path:
            self._estimate_ranges_from_data(data_path)
        else:
            # Use physiological ranges as fallback
            self.min_values = {k: v[0] for k, v in self.physiological_ranges.items()}
            self.max_values = {k: v[1] for k, v in self.physiological_ranges.items()}
    
    def _estimate_ranges_from_data(self, data_path):
        """
        Improved estimation: Reverse engineer exact min/max by analyzing
        healthy sample distributions and mapping normal values correctly
        """
        df = pd.read_csv(data_path)
        feature_cols = [col for col in df.columns if col != 'Disease']
        
        # Analyze healthy samples to understand normal value mappings
        healthy_df = df[df['Disease'] == 'Healthy'] if 'Disease' in df.columns else df
        
        # Normal physiological values (what normal raw values should be)
        normal_values = {
            'Glucose': 95,
            'Cholesterol': 180,
            'Hemoglobin': 14.5,
            'Platelets': 250000,
            'White Blood Cells': 7000,
            'Red Blood Cells': 5.0,
            'Hematocrit': 42,
            'Mean Corpuscular Volume': 90,
            'Mean Corpuscular Hemoglobin': 30,
            'Mean Corpuscular Hemoglobin Concentration': 34,
            'Insulin': 10,
            'BMI': 22,
            'Systolic Blood Pressure': 120,
            'Diastolic Blood Pressure': 80,
            'Triglycerides': 100,
            'HbA1c': 5.0,
            'LDL Cholesterol': 100,
            'HDL Cholesterol': 55,
            'ALT': 20,
            'AST': 25,
            'Heart Rate': 72,
            'Creatinine': 0.9,
            'Troponin': 0.01,
            'C-reactive Protein': 1.0
        }
        
        self.min_values = {}
        self.max_values = {}
        
        for col in feature_cols:
            # Get what normal raw value should map to (from healthy scaled mean)
            healthy_scaled_mean = healthy_df[col].mean() if len(healthy_df) > 0 else 0.5
            normal_raw = normal_values.get(col, (self.physiological_ranges[col][0] + self.physiological_ranges[col][1]) / 2)
            
            # Get physiological range
            phys_min, phys_max = self.physiological_ranges[col]
            phys_range = phys_max - phys_min
            
            # Reverse engineer: We want normal_raw to map to healthy_scaled_mean
            # scaled = (raw - min) / (max - min)
            # healthy_scaled_mean = (normal_raw - min) / (max - min)
            
            # Use extended physiological range
            extension_factor = 1.4
            extended_range = phys_range * extension_factor
            
            # Solve for min/max so that normal maps to healthy_scaled_mean
            estimated_min = normal_raw - healthy_scaled_mean * extended_range
            estimated_max = estimated_min + extended_range
            
            # Ensure reasonable bounds
            estimated_min = max(estimated_min, phys_min * 0.2)
            estimated_max = min(estimated_max, phys_max * 2.0)
            
            self.min_values[col] = estimated_min
            self.max_values[col] = estimated_max
            
        # Ensure all physiological features are present (for derived features not in CSV)
        for feature, (min_val, max_val) in self.physiological_ranges.items():
            if feature not in self.min_values:
                self.min_values[feature] = min_val
                self.max_values[feature] = max_val
    
    def scale_value(self, feature_name, raw_value):
        """
        Scale a single raw value to 0-1 range
        
        Args:
            feature_name: Name of the feature
            raw_value: Raw clinical value
            
        Returns:
            Scaled value (0-1)
        """
        if feature_name not in self.min_values:
            raise ValueError(f"Unknown feature: {feature_name}")
        
        min_val = self.min_values[feature_name]
        max_val = self.max_values[feature_name]
        
        # Min-max scaling: (x - min) / (max - min)
        scaled = (raw_value - min_val) / (max_val - min_val)
        
        # Clip to [0, 1] range
        scaled = np.clip(scaled, 0, 1)
        
        return scaled
    
    def scale_features(self, raw_features_dict):
        """
        Scale a dictionary of raw features to 0-1 range
        
        Args:
            raw_features_dict: Dictionary with feature names as keys and raw values as values
            
        Returns:
            Dictionary with scaled values
        """
        scaled_features = {}
        for feature_name, raw_value in raw_features_dict.items():
            scaled_features[feature_name] = self.scale_value(feature_name, raw_value)
        
        return scaled_features
    
    def scale_to_array(self, raw_features_dict, feature_order):
        """
        Scale features and return as numpy array in specified order
        
        Args:
            raw_features_dict: Dictionary with feature names and raw values
            feature_order: List of feature names in the order expected by model
            
        Returns:
            Numpy array of scaled features
        """
        scaled_dict = self.scale_features(raw_features_dict)
        scaled_array = np.array([scaled_dict[feature] for feature in feature_order])
        return scaled_array
    
    def get_feature_range(self, feature_name):
        """
        Get the estimated min/max range for a feature
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Tuple of (min, max) values
        """
        return (self.min_values[feature_name], self.max_values[feature_name])
    
    def save(self, filepath):
        """Save scaling bridge parameters"""
        import joblib
        joblib.dump({
            'min_values': self.min_values,
            'max_values': self.max_values,
            'physiological_ranges': self.physiological_ranges
        }, filepath)
    
    @classmethod
    def load(cls, filepath):
        """Load scaling bridge parameters"""
        data = joblib.load(filepath)
        bridge = cls()
        bridge.min_values = data['min_values']
        bridge.max_values = data['max_values']
        bridge.physiological_ranges = data['physiological_ranges']
        return bridge


def main():
    """Test the Scaling Bridge"""
    # Initialize bridge
    bridge = ScalingBridge('data/Blood_samples_dataset_balanced_2(f).csv')
    
    # Test with sample raw values
    sample_raw = {
        'Glucose': 120,  # mg/dL
        'BMI': 25.5,  # kg/m²
        'Troponin': 0.02,  # ng/mL
        'Systolic Blood Pressure': 130,  # mmHg
        'Hemoglobin': 14.5  # g/dL
    }
    
    print("Testing Scaling Bridge:")
    print("\nRaw Values:")
    for feature, value in sample_raw.items():
        print(f"  {feature}: {value}")
    
    print("\nScaled Values:")
    scaled = bridge.scale_features(sample_raw)
    for feature, value in scaled.items():
        print(f"  {feature}: {value:.4f}")
    
    # Save bridge
    import os
    os.makedirs('models', exist_ok=True)
    bridge.save('models/scaling_bridge.pkl')
    print("\n✓ Scaling Bridge saved to 'models/scaling_bridge.pkl'")
    
    return bridge

if __name__ == "__main__":
    main()

