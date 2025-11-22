#!/bin/bash
# Script to run the MediGuard AI Web Application

echo "🏥 Starting MediGuard AI Web Application..."
echo ""

# Check if model files exist
if [ ! -f "models/best_model.pkl" ]; then
    echo "⚠️  Model files not found. Training model first..."
    python3 module_a_train_model.py
    python3 module_b_scaling_bridge.py
    echo ""
fi

# Check if Flask dependencies are installed
python3 -c "import flask" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip3 install -q -r requirements.txt
    echo ""
fi

# Run the Flask application
echo "🚀 Launching Flask Web Application..."
echo "📱 Open your browser and navigate to: http://localhost:5000"
echo ""
python3 run_app.py

