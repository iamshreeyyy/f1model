#!/bin/bash

# F1 AI Backend Startup Script

echo "=== F1 AI Backend Setup ==="

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv f1_env

# Activate virtual environment
echo "Activating virtual environment..."
source f1_env/bin/activate

# Install dependencies
echo "Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Train initial model
echo "Training initial ML model..."
python model.py

# Start the API server
echo "Starting FastAPI server..."
echo "API will be available at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo ""

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
