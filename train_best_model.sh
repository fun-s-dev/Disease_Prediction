#!/bin/bash
# Script to train the model

echo "🚀 Training Model..."
echo ""

# Check if data exists
if [ ! -f "data/Blood_samples_dataset_balanced_2(f).csv" ]; then
    echo "❌ Error: Training data not found!"
    exit 1
fi

# Run training
echo "Running training pipeline..."
python3 module_a_train_model.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model training completed successfully!"
    echo ""
    echo "✓ Model is ready to use in the web application!"
else
    echo ""
    echo "❌ Training failed. Check errors above."
    exit 1
fi

