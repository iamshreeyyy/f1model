"""
Professional ML Pipeline Integration
Integrates all pipeline components with FastAPI backend
"""

import sys
import os
from pathlib import Path

# Add the current directory to the Python path
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime
import asyncio

# Import our professional pipeline components
try:
    from data_acquisition import F1DataAcquisition
    from feature_engineering import F1FeatureEngineering
    from ml_pipeline import F1MLPipeline
    from continuous_update import F1ContinuousUpdate
    from decision_support import F1DecisionSupport
    from enhanced_f1_models import EnhancedF1Models
    from model import F1PredictionModel
except ImportError as e:
    logging.warning(f"Import warning: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="F1 Professional ML Pipeline API", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3003", "http://127.0.0.1:3003"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize pipeline components
data_acquisition = F1DataAcquisition()
feature_engineering = F1FeatureEngineering()
ml_pipeline = F1MLPipeline()
continuous_update = F1ContinuousUpdate()
decision_support = F1DecisionSupport()
enhanced_models = EnhancedF1Models()
prediction_model = F1PredictionModel()

# Pydantic models for API requests
class DataAcquisitionRequest(BaseModel):
    years: List[int]
    force_refresh: bool = False

class FeatureEngineeringRequest(BaseModel):
    input_file: Optional[str] = None
    output_file: Optional[str] = None

class TrainingRequest(BaseModel):
    data_file: Optional[str] = None
    target_metric: str = "f1_score"
    cross_validation_folds: int = 5

class PredictionRequest(BaseModel):
    race_data: Dict[str, Any]
    weather: Dict[str, str]
    use_enhanced_models: bool = True

class StrategyRequest(BaseModel):
    driver: str
    circuit: str
    weather_forecast: Dict[str, Any]
    race_predictions: List[Dict[str, Any]]

# Global variables for pipeline status
pipeline_status = {
    "data_acquisition": {"status": "ready", "last_update": None},
    "feature_engineering": {"status": "ready", "last_update": None},
    "model_training": {"status": "ready", "last_update": None},
    "continuous_update": {"status": "ready", "last_update": None}
}

@app.get("/api/status")
async def get_pipeline_status():
    """Get overall pipeline status"""
    return {
        "pipeline_status": pipeline_status,
        "timestamp": datetime.now().isoformat(),
        "api_version": "2.0.0"
    }

@app.post("/api/data/acquire")
async def acquire_f1_data(request: DataAcquisitionRequest, background_tasks: BackgroundTasks):
    """
    Acquire F1 data for specified years
    """
    try:
        logger.info(f"Starting data acquisition for years: {request.years}")
        pipeline_status["data_acquisition"]["status"] = "running"
        
        # Run data acquisition in background
        def run_acquisition():
            try:
                consolidated_data = data_acquisition.acquire_multi_year_data(
                    years=request.years,
                    force_refresh=request.force_refresh
                )
                pipeline_status["data_acquisition"]["status"] = "completed"
                pipeline_status["data_acquisition"]["last_update"] = datetime.now().isoformat()
                pipeline_status["data_acquisition"]["records_acquired"] = len(consolidated_data)
                return consolidated_data
            except Exception as e:
                pipeline_status["data_acquisition"]["status"] = "error"
                pipeline_status["data_acquisition"]["error"] = str(e)
                logger.error(f"Data acquisition failed: {e}")
        
        background_tasks.add_task(run_acquisition)
        
        return {
            "message": "Data acquisition started",
            "years": request.years,
            "status": "running",
            "estimated_completion": "5-10 minutes"
        }
    
    except Exception as e:
        pipeline_status["data_acquisition"]["status"] = "error"
        pipeline_status["data_acquisition"]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Data acquisition failed: {str(e)}")

@app.post("/api/features/engineer")
async def engineer_features(request: FeatureEngineeringRequest, background_tasks: BackgroundTasks):
    """
    Engineer features from raw F1 data
    """
    try:
        logger.info("Starting feature engineering")
        pipeline_status["feature_engineering"]["status"] = "running"
        
        def run_feature_engineering():
            try:
                input_file = request.input_file or "data/f1_consolidated_data.csv"
                output_file = request.output_file or "data/f1_engineered_features.csv"
                
                engineered_features = feature_engineering.create_comprehensive_features(
                    input_file=input_file,
                    output_file=output_file
                )
                
                pipeline_status["feature_engineering"]["status"] = "completed"
                pipeline_status["feature_engineering"]["last_update"] = datetime.now().isoformat()
                pipeline_status["feature_engineering"]["features_created"] = len(engineered_features.columns)
                
                return engineered_features
            except Exception as e:
                pipeline_status["feature_engineering"]["status"] = "error"
                pipeline_status["feature_engineering"]["error"] = str(e)
                logger.error(f"Feature engineering failed: {e}")
        
        background_tasks.add_task(run_feature_engineering)
        
        return {
            "message": "Feature engineering started",
            "status": "running",
            "input_file": request.input_file or "data/f1_consolidated_data.csv",
            "output_file": request.output_file or "data/f1_engineered_features.csv"
        }
    
    except Exception as e:
        pipeline_status["feature_engineering"]["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Feature engineering failed: {str(e)}")

@app.post("/api/models/train")
async def train_models(request: TrainingRequest, background_tasks: BackgroundTasks):
    """
    Train ML models with engineered features
    """
    try:
        logger.info("Starting model training")
        pipeline_status["model_training"]["status"] = "running"
        
        def run_training():
            try:
                data_file = request.data_file or "data/f1_engineered_features.csv"
                
                training_results = ml_pipeline.train_ensemble_models(
                    data_file=data_file,
                    target_metric=request.target_metric,
                    cv_folds=request.cross_validation_folds
                )
                
                pipeline_status["model_training"]["status"] = "completed"
                pipeline_status["model_training"]["last_update"] = datetime.now().isoformat()
                pipeline_status["model_training"]["best_score"] = training_results.get('best_cv_score', 0)
                
                return training_results
            except Exception as e:
                pipeline_status["model_training"]["status"] = "error"
                pipeline_status["model_training"]["error"] = str(e)
                logger.error(f"Model training failed: {e}")
        
        background_tasks.add_task(run_training)
        
        return {
            "message": "Model training started",
            "status": "running",
            "target_metric": request.target_metric,
            "cross_validation_folds": request.cross_validation_folds,
            "estimated_completion": "15-30 minutes"
        }
    
    except Exception as e:
        pipeline_status["model_training"]["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Model training failed: {str(e)}")

@app.post("/api/predictions/professional")
async def get_professional_predictions(request: PredictionRequest):
    """
    Get predictions using professional ML pipeline
    """
    try:
        if request.use_enhanced_models:
            # Use enhanced models for prediction
            prediction_result = enhanced_models.predict_race_outcome(
                race_data=request.race_data,
                weather_conditions=request.weather
            )
        else:
            # Use standard prediction model
            prediction_result = prediction_model.predict_race_positions(
                race_data=request.race_data,
                weather=request.weather
            )
        
        return {
            "predictions": prediction_result,
            "model_type": "enhanced" if request.use_enhanced_models else "standard",
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/api/strategy/analyze")
async def analyze_race_strategy(request: StrategyRequest):
    """
    Generate comprehensive race strategy analysis
    """
    try:
        # Load historical data (in production, this would be from the pipeline)
        import pandas as pd
        try:
            historical_data = pd.read_csv("data/f1_engineered_features.csv")
        except FileNotFoundError:
            # Create mock historical data for demonstration
            historical_data = pd.DataFrame({
                'driver': [request.driver] * 5,
                'circuit_name': [request.circuit.lower()] * 5,
                'finishing_position': [3, 5, 2, 4, 1],
                'grid_position': [2, 4, 1, 3, 2]
            })
        
        strategy_report = decision_support.generate_race_strategy_report(
            driver=request.driver,
            circuit=request.circuit,
            race_predictions=request.race_predictions,
            weather_forecast=request.weather_forecast,
            historical_data=historical_data
        )
        
        return strategy_report
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Strategy analysis failed: {str(e)}")

@app.get("/api/strategy/pit-optimization/{circuit}")
async def get_pit_optimization(circuit: str, track_temp: float = 35, rain_prob: float = 0):
    """
    Get pit stop optimization for specific circuit
    """
    try:
        # Mock race predictions for demonstration
        race_predictions = [
            {'driver': 'Max Verstappen', 'predicted_position': 1},
            {'driver': 'Lando Norris', 'predicted_position': 2},
            {'driver': 'Charles Leclerc', 'predicted_position': 3}
        ]
        
        weather_conditions = {
            'track_temperature': track_temp,
            'rain_probability': rain_prob
        }
        
        pit_strategy = decision_support.predict_pit_window_optimization(
            race_predictions=race_predictions,
            circuit=circuit,
            weather_conditions=weather_conditions
        )
        
        return pit_strategy
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pit optimization failed: {str(e)}")

@app.post("/api/continuous/update")
async def trigger_continuous_update(background_tasks: BackgroundTasks):
    """
    Trigger continuous update process
    """
    try:
        logger.info("Starting continuous update process")
        pipeline_status["continuous_update"]["status"] = "running"
        
        def run_continuous_update():
            try:
                update_results = continuous_update.run_update_cycle()
                
                pipeline_status["continuous_update"]["status"] = "completed"
                pipeline_status["continuous_update"]["last_update"] = datetime.now().isoformat()
                pipeline_status["continuous_update"]["updates_applied"] = len(update_results.get('updates', []))
                
                return update_results
            except Exception as e:
                pipeline_status["continuous_update"]["status"] = "error"
                pipeline_status["continuous_update"]["error"] = str(e)
                logger.error(f"Continuous update failed: {e}")
        
        background_tasks.add_task(run_continuous_update)
        
        return {
            "message": "Continuous update process started",
            "status": "running",
            "estimated_completion": "10-20 minutes"
        }
    
    except Exception as e:
        pipeline_status["continuous_update"]["status"] = "error"
        raise HTTPException(status_code=500, detail=f"Continuous update failed: {str(e)}")

@app.get("/api/pipeline/health")
async def pipeline_health_check():
    """
    Comprehensive pipeline health check
    """
    health_status = {
        "overall_status": "healthy",
        "components": {},
        "data_files": {},
        "timestamp": datetime.now().isoformat()
    }
    
    # Check component availability
    components = {
        "data_acquisition": data_acquisition,
        "feature_engineering": feature_engineering,
        "ml_pipeline": ml_pipeline,
        "continuous_update": continuous_update,
        "decision_support": decision_support,
        "enhanced_models": enhanced_models
    }
    
    for name, component in components.items():
        try:
            # Simple check - component is initialized
            health_status["components"][name] = {
                "status": "available",
                "type": type(component).__name__
            }
        except Exception as e:
            health_status["components"][name] = {
                "status": "error",
                "error": str(e)
            }
            health_status["overall_status"] = "degraded"
    
    # Check data files
    data_files = [
        "data/f1_consolidated_data.csv",
        "data/f1_engineered_features.csv",
        "data/models/f1_model.pkl"
    ]
    
    for file_path in data_files:
        try:
            if os.path.exists(file_path):
                file_size = os.path.getsize(file_path)
                health_status["data_files"][file_path] = {
                    "status": "available",
                    "size_bytes": file_size,
                    "size_mb": round(file_size / 1024 / 1024, 2)
                }
            else:
                health_status["data_files"][file_path] = {
                    "status": "missing"
                }
        except Exception as e:
            health_status["data_files"][file_path] = {
                "status": "error",
                "error": str(e)
            }
    
    return health_status

@app.get("/api/pipeline/metrics")
async def get_pipeline_metrics():
    """
    Get pipeline performance metrics
    """
    try:
        # Load any available training metrics
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "model_performance": {},
            "data_quality": {},
            "pipeline_efficiency": {}
        }
        
        # Try to load model metrics if available
        try:
            with open("data/models/training_metrics.json", "r") as f:
                metrics["model_performance"] = json.load(f)
        except FileNotFoundError:
            metrics["model_performance"] = {"status": "no_metrics_available"}
        
        # Data quality metrics
        try:
            if os.path.exists("data/f1_engineered_features.csv"):
                import pandas as pd
                df = pd.read_csv("data/f1_engineered_features.csv")
                metrics["data_quality"] = {
                    "total_records": len(df),
                    "total_features": len(df.columns),
                    "missing_data_percentage": (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
                }
        except Exception as e:
            metrics["data_quality"] = {"error": str(e)}
        
        # Pipeline efficiency
        metrics["pipeline_efficiency"] = pipeline_status
        
        return metrics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "F1 Professional ML Pipeline"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
