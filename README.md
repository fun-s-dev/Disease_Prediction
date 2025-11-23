# 🏥 MediGuard AI: Intelligent Disease Prediction System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/flask-3.0+-green.svg)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An AI-powered medical triage assistant that analyzes 24 blood test parameters to predict the likelihood of multiple diseases including Heart Disease, Diabetes, Anemia, Thalassemia, and Thrombocytopenia.

## ✨ Features

### 🌐 Web Application (Flask)
- **User Authentication**: Secure registration and login system with password hashing
- **Interactive Dashboard**: Real-time disease prediction with visual feedback
- **Prediction History**: Complete record of all predictions with detailed views
- **SHAP Explainability**: AI-powered explanations showing which features influenced predictions
- **Data Quality Checks**: Automatic detection of outliers and data quality issues
- **Blockchain Audit Trail**: Immutable logging of all predictions for medical compliance
- **Responsive Design**: Modern, mobile-friendly interface with Bootstrap

### 🤖 Machine Learning
- **Multi-Class Classification**: XGBoost + Random Forest ensemble model
- **Optimized for Medical Use**: High recall (sensitivity) to minimize dangerous false negatives
- **SMOTE Balancing**: Handles class imbalance in medical datasets
- **Feature Engineering**: Clinically relevant derived features (LDL/HDL ratio, MAP, etc.)
- **Model Performance**: 93.3% accuracy with 72% macro recall on test set

### 📊 Analytics & Visualization
- Feature importance analysis with SHAP values
- Probability distributions for all disease classes
- Confusion matrix and classification reports
- Real-time risk level indicators
- Historical trend analysis

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Gauraangst/Disease_Prediction.git
cd Disease_Prediction
```

2. **Create virtual environment** (recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Train the model**
```bash
python3 module_a_train_model.py
```
This will:
- Load and preprocess training data
- Train XGBoost and Random Forest models
- Generate ensemble model with best recall performance
- Save models to `models/` directory

5. **Run the web application**
```bash
python3 app.py
```
Navigate to `http://localhost:5000` in your browser.

## 📁 Project Structure

```
├── app.py                          # Flask web application
├── models.py                       # Database models (User, Prediction)
├── module_a_train_model.py         # Model training pipeline
├── module_b_scaling_bridge.py      # Feature scaling and normalization
├── anomaly_detector.py             # Data quality and outlier detection
├── evaluation.py                   # Model evaluation and metrics
├── migrate_db.py                   # Database migration utility
├── data/
│   ├── Blood_samples_dataset_balanced_2(f).csv
│   └── blood_samples_dataset_test.csv
├── models/                         # Trained model files (generated)
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   ├── scaling_bridge.pkl
│   └── shap_explainer.pkl
├── templates/                      # HTML templates
│   ├── base.html
│   ├── dashboard.html
│   ├── reports.html
│   └── ...
└── static/                         # CSS, JavaScript, images
    ├── css/style.css
    └── js/main.js
```

## 🩺 Blood Test Parameters

The system analyzes 24 clinical parameters:

**Metabolic Markers:**
- Glucose, Insulin, HbA1c, BMI

**Blood Cell Analysis:**
- Hemoglobin, Platelets, White Blood Cells, Red Blood Cells
- Hematocrit, MCV, MCH, MCHC

**Cardiovascular:**
- Systolic/Diastolic Blood Pressure, Heart Rate
- Cholesterol, Triglycerides, LDL, HDL
- Troponin, C-reactive Protein

**Organ Function:**
- ALT, AST, Creatinine

## 🎯 Supported Diseases

1. **Diabetes** - Glucose metabolism disorder
2. **Heart Disease** - Cardiovascular conditions
3. **Anemia** - Low red blood cell count
4. **Thalassemia** - Inherited blood disorder
5. **Thrombocytopenia** - Low platelet count
6. **Healthy** - No disease detected

## 📊 Model Performance

Evaluated on independent test set:

| Metric | Score |
|--------|-------|
| **Accuracy** | 93.3% |
| **Macro Recall** | 71.9% |
| **Weighted Recall** | 93.3% |
| **Macro F1-Score** | 75.0% |

Per-class recall (sensitivity):
- Healthy: 98.5%
- Diabetes: 93.5%
- Anemia: 85.0%
- Thalassemia: 66.7%
- Heart Disease: 37.5%
- Thrombocytopenia: 50.0%

*High recall minimizes false negatives, critical for medical triage.*

## 🔧 Usage

### Web Interface

1. **Register/Login**: Create an account or log in
2. **Dashboard**: Enter patient blood test results
3. **Predict**: Click "Predict Disease" to get instant results
4. **View Results**: See prediction, confidence, risk level, and explanations
5. **Reports**: Access prediction history and detailed reports

### Programmatic Use

```python
from module_b_scaling_bridge import ScalingBridge
import joblib

# Load model and scaler
model = joblib.load('models/best_model.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')
scaling_bridge = ScalingBridge.load('models/scaling_bridge.pkl')

# Prepare raw features
raw_features = {
    'Glucose': 120,
    'Insulin': 15,
    'BMI': 25,
    # ... all 24 parameters
}

# Scale and predict
scaled_features = scaling_bridge.scale_to_array(raw_features, feature_names)
prediction = model.predict(scaled_features.reshape(1, -1))[0]
disease = label_encoder.inverse_transform([prediction])[0]
```

## 🔐 Security Features

- Password hashing with Werkzeug
- Flask-Login session management
- CSRF protection
- SQL injection prevention with SQLAlchemy ORM
- Blockchain-style audit logging

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It should NOT be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of qualified health providers with any questions regarding medical conditions.

## 👥 Authors

**Gauraang Thakkar** - [Gauraangst](https://github.com/Gauraangst)
**Aryan Tanna** - [Aryan](https://github.com/Aryan-Tanna)
**Anjali Sinha** - [Anjali](https://github.com/fun-s-dev)
**Parth Shah** - [Parth](https://github.com/parth-shah23)

## 🙏 Acknowledgments

- XGBoost and scikit-learn communities
- Flask and Bootstrap frameworks
- SHAP library for model explainability
- Medical domain experts for parameter validation

---

Made with ❤️ for better healthcare through AI
