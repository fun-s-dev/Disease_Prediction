#!/bin/bash
# Script to run the MediGuard AI Dashboard

echo "🏥 Starting MediGuard AI Dashboard..."
echo ""

# Check if model files exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️  Model files not found. Training model first..."
    python3 module_a_train_model.py
    python3 module_b_scaling_bridge.py
    echo ""
fi

# Run the dashboard
echo "🚀 Launching Streamlit Dashboard..."
streamlit run module_c_dashboard.py

