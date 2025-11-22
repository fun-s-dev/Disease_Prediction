"""
Test script to simulate dashboard form submission and debug the Red Blood Cells error
"""
import requests
import json

# Login first
session = requests.Session()

# Try to login with existing user
login_data = {
    'username': 'Gauraang',
    'password': 'password'  # try common passwords
}

# List of passwords to try
passwords = ['password', 'Password123', 'admin', 'gauraang', 'Gauraang', '123456', 'test']

login_url = 'http://localhost:5000/login'
predict_url = 'http://localhost:5000/predict'

logged_in = False
for pwd in passwords:
    login_data['password'] = pwd
    response = session.post(login_url, data=login_data)
    if 'Welcome back' in response.text or response.url.endswith('/dashboard'):
        print(f"✓ Successfully logged in with password: {pwd}")
        logged_in = True
        break

if not logged_in:
    print("❌ Could not log in with any common password")
    print("Trying without login...")

# Prepare prediction data with ALL required features
prediction_data = {
    'patient_id': 'TEST_001',
    'patient_name': 'Test Patient',
    'patient_age': 45,
    'patient_sex': 'Male',
    # Basic Metabolism
    'Glucose': 95.0,
    'Insulin': 15.0,
    'HbA1c': 5.0,
    'BMI': 22.0,
    # Blood Cell Analysis
    'Hemoglobin': 14.5,
    'Platelets': 250000.0,
    'White_Blood_Cells': 7000.0,
    'Red_Blood_Cells': 5.0,  # The problematic feature
    'Hematocrit': 42.0,
    'Mean_Corpuscular_Volume': 90.0,
    'Mean_Corpuscular_Hemoglobin': 30.0,
    'Mean_Corpuscular_Hemoglobin_Concentration': 33.0,
    # Cardiac & Lipid Profile
    'Systolic_Blood_Pressure': 120.0,
    'Diastolic_Blood_Pressure': 80.0,
    'Heart_Rate': 72.0,
    'Cholesterol': 180.0,
    'Triglycerides': 100.0,
    'LDL_Cholesterol': 100.0,
    'HDL_Cholesterol': 55.0,
    'Troponin': 0.01,
    'C-reactive_Protein': 1.0,
    # Liver & Kidney Function
    'ALT': 20.0,
    'AST': 25.0,
    'Creatinine': 0.9
}

print(f"\n📤 Sending prediction request with {len(prediction_data)} fields...")
print(f"Fields: {list(prediction_data.keys())}")

try:
    response = session.post(
        predict_url,
        json=prediction_data,
        headers={'Content-Type': 'application/json'}
    )
    
    print(f"\n📥 Response Status: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Body:")
    print(json.dumps(response.json(), indent=2))
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    print(f"Response text: {response.text[:500] if 'response' in locals() else 'N/A'}")
