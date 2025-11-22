# MediGuard AI: Intelligent Triage Assistant

An intelligent triage assistant that analyzes 24 pre-scaled blood test parameters to predict the likelihood of multiple diseases (Heart Disease, Diabetes, Anemia, Thalassemia, Thrombocytopenia, Healthy).

## Project Structure

```
.
├── data/
│   ├── Blood_samples_dataset_balanced_2(f).csv  # Training dataset
│   └── blood_samples_dataset_test.csv           # Test dataset
├── models/                                       # Generated model files
│   ├── best_model.pkl
│   ├── label_encoder.pkl
│   ├── feature_names.pkl
│   └── scaling_bridge.pkl
├── templates/                                   # HTML templates
│   ├── base.html
│   ├── index.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── reports.html
│   ├── report_detail.html
│   └── profile.html
├── static/                                      # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
├── app.py                                       # Flask web application
├── models.py                                    # Database models
├── module_a_train_model.py                      # Module A: Model Training
├── module_b_scaling_bridge.py                  # Module B: Scaling Bridge
├── module_c_dashboard.py                       # Module C: Streamlit Dashboard
├── evaluation.py                               # Evaluation Module
├── run_app.py                                  # Flask app runner
├── run_webapp.sh                               # Web app startup script
├── requirements.txt                            # Dependencies
└── README.md                                   # This file
```

## Features

### Web Application (Flask)
- **User Authentication**: Secure registration and login system
- **Landing Page**: Professional homepage with feature showcase
- **Dashboard**: Interactive prediction interface with real-time results
- **Reports**: Complete prediction history with detailed views
- **User Profile**: Account management and statistics
- **Database**: SQLite database for persistent data storage
- **Responsive Design**: Modern, mobile-friendly UI

### Module A: Model Training
- Multi-class classification using XGBoost and Random Forest
- Optimized for high Recall (Sensitivity) to minimize False Negatives
- Handles class imbalance
- Saves best model based on recall performance

### Module B: Scaling Bridge
- Converts raw clinical values to 0-1 scaled format
- Uses physiological ranges for accurate scaling
- Estimates min/max from training data
- Handles all 24 blood test parameters

### Module C: Dashboard (Streamlit Alternative)
- Interactive Streamlit web application
- Real-time disease prediction
- SHAP-based explainability
- Feature importance visualization
- Risk indicators
- **Bonus**: Data quality detection (outlier detection)
- **Bonus**: Blockchain audit trail for predictions

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

Train the model using:
```bash
python3 module_a_train_model.py
```
Or use the script:
```bash
./train_best_model.sh
```

This will:
- Load and prepare the training data
- Train XGBoost and Random Forest models
- Select the best model based on recall
- Save model files to `models/` directory

### Step 2: Test Scaling Bridge (Optional)
```bash
python module_b_scaling_bridge.py
```

### Step 3: Run the Web Application (Recommended)

**Option A: Full-Featured Web Application (Flask)**
```bash
python3 run_app.py
```
Or use the script:
```bash
./run_webapp.sh
```

The web application will start at `http://localhost:5000`. Features include:
- **User Authentication**: Register and login system
- **Landing Page**: Beautiful homepage with feature overview
- **Dashboard**: Interactive prediction form with real-time results
- **Reports**: View and manage all your predictions
- **Profile**: User profile and statistics
- **Database**: SQLite database for users and predictions
- All features from the Streamlit dashboard

**Option B: Streamlit Dashboard (Alternative)**
```bash
streamlit run module_c_dashboard.py
```

The Streamlit dashboard provides:
- Enter raw clinical values in the sidebar
- Get disease predictions with confidence scores
- View feature importance and explainability
- See risk indicators
- Check data quality warnings
- View blockchain audit trail

### Step 4: Evaluate Performance
```bash
python evaluation.py
```

This generates a comprehensive evaluation report including:
- Model performance metrics (prioritizing Recall)
- Per-class recall scores
- Confusion matrix
- Scaling bridge validation

## Dataset

The dataset contains 24 blood test parameters:
- Glucose, Cholesterol, Hemoglobin, Platelets
- White Blood Cells, Red Blood Cells, Hematocrit
- Mean Corpuscular Volume, Mean Corpuscular Hemoglobin, Mean Corpuscular Hemoglobin Concentration
- Insulin, BMI, Systolic/Diastolic Blood Pressure
- Triglycerides, HbA1c, LDL/HDL Cholesterol
- ALT, AST, Heart Rate, Creatinine, Troponin, C-reactive Protein

**Disease Labels:**
- Healthy
- Diabetes
- Thalassemia
- Anemia
- Thrombocytopenia

## Model Performance

The model is optimized for **high Recall (Sensitivity)** to minimize dangerous False Negatives in medical triage scenarios. Key metrics:
- **Macro Recall**: Primary metric for evaluation
- **Weighted Recall**: Overall recall across all classes
- **Per-class Recall**: Individual disease sensitivity

## Scaling Bridge

The Scaling Bridge converts raw clinical values (e.g., Glucose: 120 mg/dL) to the 0-1 scaled format required by the model. It uses:
- Physiological ranges for each parameter
- Min-max scaling: `(value - min) / (max - min)`
- Automatic clipping to [0, 1] range

## Bonus Features

### 1. Blockchain Logging
- Creates immutable audit trail for each prediction
- Hashes patient ID, prediction, timestamp, and features
- Provides non-repudiable medical record

### 2. Data Quality Detection
- Detects extreme outliers outside physiological ranges
- Warns about values outside expected ranges
- Helps identify data entry errors

## Technical Details

- **Framework**: Streamlit for dashboard
- **ML Models**: XGBoost, Random Forest
- **Explainability**: SHAP (SHapley Additive exPlanations)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Evaluation**: scikit-learn metrics

## Output Files

After running the modules, you'll have:
- `models/best_model.pkl`: Trained model
- `models/label_encoder.pkl`: Label encoder
- `models/feature_names.pkl`: Feature names
- `models/scaling_bridge.pkl`: Scaling bridge
- `evaluation_report.txt`: Performance evaluation
- `evaluation_confusion_matrix.png`: Confusion matrix visualization

## License

This project is for educational purposes.

## Author

MediGuard AI Development Team

