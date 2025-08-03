#!/bin/bash

# F1 Professional ML Pipeline Startup Script
# This script starts the enhanced F1 AI backend with professional ML capabilities

echo "🏎️  Starting F1 Professional ML Pipeline..."
echo "==========================================="

# Check if we're in the correct directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found. Please run this script from the f1-ai-backend directory."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "f1_env" ]; then
    echo "❌ Error: Virtual environment 'f1_env' not found."
    echo "Please create the virtual environment first:"
    echo "python -m venv f1_env"
    echo "source f1_env/bin/activate"
    echo "pip install -r requirements.txt"
    exit 1
fi

echo "🔧 Activating virtual environment..."
source f1_env/bin/activate

echo "📦 Installing/updating dependencies..."
pip install -q pandas numpy scikit-learn xgboost requests fastapi uvicorn aiofiles python-multipart

echo "📁 Creating necessary directories..."
mkdir -p data/cache
mkdir -p data/models
mkdir -p logs

echo "🧹 Cleaning up old cache files..."
rm -f data/cache/*.tmp 2>/dev/null || true

echo "🔍 Checking component availability..."
python -c "
try:
    from data_acquisition import F1DataAcquisition
    from feature_engineering import F1FeatureEngineering
    from ml_pipeline import F1MLPipeline
    from continuous_update import F1ContinuousUpdate
    from decision_support import F1DecisionSupport
    from enhanced_f1_models import EnhancedF1Models
    print('✅ All professional pipeline components available')
except ImportError as e:
    print(f'⚠️  Some components not available: {e}')
    print('   Pipeline will run in basic mode')
"

echo "🚀 Starting FastAPI server with professional ML pipeline..."
echo "📡 API will be available at: http://localhost:8000"
echo "📖 API documentation at: http://localhost:8000/docs"
echo "🔄 Professional pipeline status: http://localhost:8000/api/professional/status"
echo ""
echo "Professional Pipeline Endpoints:"
echo "  • Data Acquisition: POST /api/professional/data/acquire"
echo "  • Strategy Analysis: POST /api/professional/strategy/analyze"
echo "  • Pit Optimization: GET /api/professional/strategy/pit-optimization/{circuit}"
echo "  • Enhanced Predictions: POST /api/professional/predictions"
echo "  • Continuous Updates: POST /api/professional/continuous/update"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=========================================="

# Start the server
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload --log-level info
