# 🚀 How to Start MediGuard AI

## Quick Start

1. **Make sure dependencies are installed:**
   ```bash
   python3 check_dependencies.py
   ```
   If any are missing, install them:
   ```bash
   pip3 install -r requirements.txt
   ```

2. **Run the application:**
   ```bash
   python3 run_app.py
   ```

3. **Open in browser:**
   - Navigate to: **http://localhost:5001**
   - (Port 5001 is used to avoid conflicts with macOS AirPlay on port 5000)

## Troubleshooting

### If you see "ModuleNotFoundError: No module named 'flask'"

**Solution 1:** Install Flask and dependencies:
```bash
pip3 install flask flask-login flask-sqlalchemy werkzeug
```

**Solution 2:** Install all requirements:
```bash
pip3 install -r requirements.txt
```

**Solution 3:** If using a virtual environment, make sure it's activated:
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
pip install -r requirements.txt
```

### If port 5001 is in use

Edit `run_app.py` and change the port number:
```python
app.run(debug=True, host='0.0.0.0', port=5002)  # Change 5001 to 5002 or any available port
```

### If models are not found

Train the models first:
```bash
python3 module_a_train_model.py
python3 module_b_scaling_bridge.py
```

## First Time Setup

1. Register a new account at http://localhost:5001/register
2. Login with your credentials
3. Go to Dashboard to make predictions
4. View Reports to see your prediction history

## Need Help?

- Check `QUICKSTART.md` for detailed instructions
- Check `README.md` for full documentation
- Run `python3 check_dependencies.py` to verify all dependencies

