from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import json
import os
from datetime import datetime, timedelta
import asyncio
import aiofiles
from model import F1PredictionModel

# Import professional pipeline components
try:
    from data_acquisition import F1DataAcquisition
    from feature_engineering import F1FeatureEngineering
    from ml_pipeline import F1MLPipeline
    from continuous_update import F1ContinuousUpdate
    from decision_support import F1DecisionSupport
    from enhanced_f1_models import EnhancedF1PredictionModels as EnhancedF1Models
except ImportError as e:
    print(f"Warning: Professional pipeline components not available: {e}")

# Initialize FastAPI app
app = FastAPI(title="F1 AI Prediction API - Professional Edition", version="2.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", 
        "http://localhost:3001", 
        "http://localhost:3002", 
        "http://localhost:3003",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "http://127.0.0.1:3002",
        "http://127.0.0.1:3003"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ML model and professional pipeline components
ml_model = F1PredictionModel()

# Initialize professional pipeline components if available
try:
    data_acquisition = F1DataAcquisition()
    feature_engineering = F1FeatureEngineering()
    ml_pipeline = F1MLPipeline()
    continuous_update = F1ContinuousUpdate()
    decision_support = F1DecisionSupport()
    enhanced_models = EnhancedF1Models()
    PROFESSIONAL_PIPELINE_AVAILABLE = True
except:
    PROFESSIONAL_PIPELINE_AVAILABLE = False
    print("Professional pipeline components not available - running in basic mode")

# Data storage paths
DATA_DIR = "data"
PREDICTIONS_FILE = f"{DATA_DIR}/predictions.json"
RACE_RESULTS_FILE = f"{DATA_DIR}/race_results.json"
DRIVER_STATS_FILE = f"{DATA_DIR}/driver_stats.json"
CHAMPIONSHIPS_FILE = f"{DATA_DIR}/championships.json"

# Professional pipeline status tracking
pipeline_status = {
    "data_acquisition": {"status": "ready", "last_update": None},
    "feature_engineering": {"status": "ready", "last_update": None},
    "model_training": {"status": "ready", "last_update": None},
    "continuous_update": {"status": "ready", "last_update": None}
}

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Pydantic models
class RacePredictionRequest(BaseModel):
    drivers: List[str]
    circuit: str
    weather: Optional[Dict[str, Any]] = None

class DriverStats(BaseModel):
    driver: str
    team: str
    points: int
    wins: int
    podiums: int
    pole_positions: int
    fastest_laps: int
    dnfs: int

class RaceResult(BaseModel):
    race_name: str
    date: str
    circuit: str
    winner: str
    podium: List[str]
    fastest_lap: str
    results: List[Dict[str, Any]]

# Utility functions
async def load_json_file(file_path: str) -> Dict:
    """Load JSON file asynchronously"""
    try:
        async with aiofiles.open(file_path, 'r') as f:
            content = await f.read()
            return json.loads(content)
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

async def save_json_file(file_path: str, data: Dict):
    """Save JSON file asynchronously"""
    async with aiofiles.open(file_path, 'w') as f:
        await f.write(json.dumps(data, indent=2, default=str))

def initialize_sample_data():
    """Initialize sample data files"""
    
    # Sample driver stats
    sample_driver_stats = {
        "last_updated": datetime.now().isoformat(),
        "drivers": [
            {
                "driver": "Max Verstappen",
                "team": "Red Bull Racing",
                "points": 575,
                "wins": 19,
                "podiums": 21,
                "pole_positions": 7,
                "fastest_laps": 5,
                "dnfs": 1,
                "position": 1
            },
            {
                "driver": "Sergio Pérez",
                "team": "Red Bull Racing",
                "points": 285,
                "wins": 2,
                "podiums": 8,
                "pole_positions": 1,
                "fastest_laps": 1,
                "dnfs": 3,
                "position": 2
            },
            {
                "driver": "Lewis Hamilton",
                "team": "Mercedes",
                "points": 234,
                "wins": 1,
                "podiums": 6,
                "pole_positions": 2,
                "fastest_laps": 2,
                "dnfs": 1,
                "position": 3
            },
            {
                "driver": "Fernando Alonso",
                "team": "Aston Martin",
                "points": 206,
                "wins": 0,
                "podiums": 8,
                "pole_positions": 0,
                "fastest_laps": 1,
                "dnfs": 2,
                "position": 4
            },
            {
                "driver": "Charles Leclerc",
                "team": "Ferrari",
                "points": 206,
                "wins": 1,
                "podiums": 4,
                "pole_positions": 3,
                "fastest_laps": 3,
                "dnfs": 4,
                "position": 5
            },
            {
                "driver": "Lando Norris",
                "team": "McLaren",
                "points": 205,
                "wins": 0,
                "podiums": 7,
                "pole_positions": 1,
                "fastest_laps": 2,
                "dnfs": 1,
                "position": 6
            }
        ]
    }
    
    # Sample race results
    sample_race_results = {
        "last_updated": datetime.now().isoformat(),
        "races": [
            {
                "race_name": "Abu Dhabi Grand Prix",
                "date": "2024-12-08",
                "circuit": "Yas Marina Circuit",
                "winner": "Max Verstappen",
                "podium": ["Max Verstappen", "Lando Norris", "Charles Leclerc"],
                "fastest_lap": "Oscar Piastri",
                "results": [
                    {"position": 1, "driver": "Max Verstappen", "team": "Red Bull Racing", "time": "1:26:38.736", "points": 25},
                    {"position": 2, "driver": "Lando Norris", "team": "McLaren", "time": "+7.456", "points": 18},
                    {"position": 3, "driver": "Charles Leclerc", "team": "Ferrari", "time": "+31.928", "points": 15}
                ]
            },
            {
                "race_name": "Las Vegas Grand Prix",
                "date": "2024-11-24",
                "circuit": "Las Vegas Street Circuit",
                "winner": "George Russell",
                "podium": ["George Russell", "Lewis Hamilton", "Carlos Sainz"],
                "fastest_lap": "Lando Norris",
                "results": [
                    {"position": 1, "driver": "George Russell", "team": "Mercedes", "time": "1:22:05.969", "points": 26},
                    {"position": 2, "driver": "Lewis Hamilton", "team": "Mercedes", "time": "+7.313", "points": 18},
                    {"position": 3, "driver": "Carlos Sainz", "team": "Ferrari", "time": "+11.906", "points": 15}
                ]
            }
        ]
    }
    
    # Sample championships
    sample_championships = {
        "last_updated": datetime.now().isoformat(),
        "drivers_championship": {
            "year": 2024,
            "standings": [
                {"position": 1, "driver": "Max Verstappen", "team": "Red Bull Racing", "points": 575},
                {"position": 2, "driver": "Lando Norris", "team": "McLaren", "points": 374},
                {"position": 3, "driver": "Charles Leclerc", "team": "Ferrari", "points": 356},
                {"position": 4, "driver": "Oscar Piastri", "team": "McLaren", "points": 292},
                {"position": 5, "driver": "Carlos Sainz", "team": "Ferrari", "points": 290}
            ]
        },
        "constructors_championship": {
            "year": 2024,
            "standings": [
                {"position": 1, "team": "McLaren", "points": 666},
                {"position": 2, "team": "Ferrari", "points": 652},
                {"position": 3, "team": "Red Bull Racing", "points": 589},
                {"position": 4, "team": "Mercedes", "points": 468},
                {"position": 5, "team": "Aston Martin", "points": 86}
            ]
        }
    }
    
    # Sample predictions
    sample_predictions = {
        "last_updated": datetime.now().isoformat(),
        "next_race": {
            "race_name": "Bahrain Grand Prix",
            "date": "2025-03-02",
            "circuit": "Bahrain International Circuit",
            "predictions": [
                {"driver": "Max Verstappen", "position": 1, "confidence": 94, "change": 2},
                {"driver": "Lando Norris", "position": 2, "confidence": 87, "change": -1},
                {"driver": "Charles Leclerc", "position": 3, "confidence": 82, "change": 3},
                {"driver": "Oscar Piastri", "position": 4, "confidence": 78, "change": 1},
                {"driver": "Carlos Sainz", "position": 5, "confidence": 74, "change": -2},
                {"driver": "George Russell", "position": 6, "confidence": 71, "change": 0}
            ]
        },
        "recent_activity": [
            {
                "type": "prediction",
                "message": "Model accuracy improved to 94.2%",
                "timestamp": (datetime.now() - timedelta(hours=2)).isoformat()
            },
            {
                "type": "training",
                "message": "Model retrained with latest race data",
                "timestamp": (datetime.now() - timedelta(days=1)).isoformat()
            },
            {
                "type": "prediction",
                "message": "Race predictions updated for Bahrain GP",
                "timestamp": (datetime.now() - timedelta(days=2)).isoformat()
            }
        ]
    }
    
    # Save sample data
    with open(DRIVER_STATS_FILE, 'w') as f:
        json.dump(sample_driver_stats, f, indent=2)
    
    with open(RACE_RESULTS_FILE, 'w') as f:
        json.dump(sample_race_results, f, indent=2)
    
    with open(CHAMPIONSHIPS_FILE, 'w') as f:
        json.dump(sample_championships, f, indent=2)
    
    with open(PREDICTIONS_FILE, 'w') as f:
        json.dump(sample_predictions, f, indent=2)

# Initialize sample data on startup
initialize_sample_data()

# Background task to train model
async def train_model_background():
    """Train the ML model in the background"""
    try:
        print("Training ML model...")
        data = ml_model.fetch_season_data(2024)
        ml_model.train_models(data)
        print("Model training completed!")
    except Exception as e:
        print(f"Error training model: {e}")

# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("Starting F1 AI Prediction API...")
    
    # Try to load existing model
    if not ml_model.load_models():
        # If no model exists, train in background
        asyncio.create_task(train_model_background())

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "F1 AI Prediction API", "version": "1.0.0", "status": "online"}

@app.get("/api/model-stats")
async def get_model_stats():
    """Get ML model statistics"""
    stats = ml_model.get_model_stats()
    return {
        "model_accuracy": f"{stats['accuracy']}%",
        "predictions_made": stats['predictions_made'],
        "success_rate": f"{stats['success_rate']}%",
        "model_version": stats['model_version'],
        "last_updated": stats['last_updated']
    }

@app.get("/api/predictions")
async def get_predictions():
    """Get race predictions"""
    data = await load_json_file(PREDICTIONS_FILE)
    return data

@app.get("/api/race-results")
async def get_race_results():
    """Get race results"""
    data = await load_json_file(RACE_RESULTS_FILE)
    return data

@app.get("/api/driver-stats")
async def get_driver_stats():
    """Get driver statistics"""
    data = await load_json_file(DRIVER_STATS_FILE)
    return data

@app.get("/api/championships")
async def get_championships():
    """Get championship standings"""
    data = await load_json_file(CHAMPIONSHIPS_FILE)
    return data

@app.post("/api/predict-race-enhanced")
async def predict_race_enhanced(request: RacePredictionRequest):
    """Generate enhanced F1 race predictions using optimized ML models"""
    try:
        # Get enhanced classification predictions
        enhanced_predictions = ml_model.predict_race_classification(
            drivers=request.drivers,
            circuit=request.circuit,
            weather=request.weather
        )
        
        # Create response with just enhanced predictions for now
        combined_results = {
            "enhanced_predictions": enhanced_predictions,
            "model_performance": {
                "ensemble_models": ["XGBoost", "Random Forest", "Gradient Boosting", "Logistic Regression", "SVM"],
                "optimization_target": "F1-Score Maximization",
                "prediction_categories": {
                    "podium": "Top 3 finish probability",
                    "points_finish": "Top 10 finish probability", 
                    "winner": "Race win probability",
                    "dnf": "Did not finish probability"
                }
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(content=combined_results)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhanced prediction failed: {str(e)}")

@app.get("/api/model-comparison")
async def get_model_comparison():
    """Compare different F1 prediction models and their strengths"""
    return JSONResponse(content={
        "model_comparison": {
            "xgboost": {
                "strengths": [
                    "Direct F1-score optimization through eval_metric parameter",
                    "Excellent handling of imbalanced F1 race data",
                    "Built-in feature importance for driver/circuit analysis",
                    "Scale_pos_weight for handling rare events (wins, podiums)"
                ],
                "f1_racing_applications": [
                    "Podium finish prediction",
                    "DNF (reliability) prediction", 
                    "Weather impact on performance",
                    "Grid position advantage analysis"
                ],
                "optimization_features": [
                    "Custom evaluation metrics",
                    "Early stopping based on F1 score",
                    "Hyperparameter tuning for F1 maximization"
                ]
            },
            "random_forest": {
                "strengths": [
                    "Class balancing with class_weight='balanced'",
                    "Robust to outliers in race data",
                    "Feature importance ranking for F1 factors",
                    "Ensemble nature reduces overfitting"
                ],
                "f1_racing_applications": [
                    "Driver consistency analysis",
                    "Circuit-specific performance patterns",
                    "Multi-factor race outcome prediction",
                    "Team performance evaluation"
                ],
                "optimization_features": [
                    "Bootstrap sampling for better generalization",
                    "Out-of-bag scoring for model selection",
                    "Parallel training for efficiency"
                ]
            },
            "gradient_boosting": {
                "strengths": [
                    "Sequential learning corrects prediction errors",
                    "Custom loss functions for F1 optimization",
                    "Excellent for complex F1 race dynamics",
                    "Adaptive learning rates"
                ],
                "f1_racing_applications": [
                    "Race strategy optimization",
                    "Tire degradation impact prediction",
                    "Safety car probability analysis",
                    "Championship points scenarios"
                ],
                "optimization_features": [
                    "Learning rate scheduling",
                    "Early stopping mechanisms",
                    "Subsample optimization"
                ]
            },
            "logistic_regression": {
                "strengths": [
                    "Well-calibrated probability outputs",
                    "Threshold optimization for F1 maximization",
                    "Fast training and inference",
                    "Interpretable coefficients"
                ],
                "f1_racing_applications": [
                    "Binary race outcome prediction",
                    "Qualifying vs race performance",
                    "Points finish probability",
                    "Head-to-head driver comparisons"
                ],
                "optimization_features": [
                    "Class weighting for imbalanced data",
                    "Regularization (L1/L2) for feature selection",
                    "Threshold tuning for optimal F1 score"
                ]
            },
            "svm": {
                "strengths": [
                    "Kernel flexibility for complex F1 patterns",
                    "Robust to high-dimensional feature spaces",
                    "Memory efficient for large datasets",
                    "Class weight handling for rare events"
                ],
                "f1_racing_applications": [
                    "Non-linear race pattern recognition",
                    "Driver style classification",
                    "Circuit characteristic clustering",
                    "Performance anomaly detection"
                ],
                "optimization_features": [
                    "Multiple kernel options (RBF, polynomial, linear)",
                    "C parameter tuning for F1 optimization",
                    "Gamma optimization for kernel performance"
                ]
            }
        },
        "ensemble_advantages": {
            "why_ensemble_works": [
                "Combines strengths of different algorithms",
                "Reduces individual model biases",
                "Better generalization to unseen race scenarios",
                "Majority voting improves prediction confidence"
            ],
            "f1_racing_benefits": [
                "Handles different types of F1 prediction tasks",
                "Adapts to changing regulations and car performance",
                "Robust to driver transfers and team changes",
                "Accounts for various weather and track conditions"
            ]
        },
        "performance_metrics": {
            "f1_score_focus": "Optimized for precision and recall balance",
            "classification_targets": [
                "Podium finishes (top 3)",
                "Points finishes (top 10)",
                "Race winners",
                "DNF predictions"
            ],
            "evaluation_approach": "5-fold cross-validation with F1 scoring"
        }
    })

@app.post("/api/predict-race")
async def predict_race(request: RacePredictionRequest):
    """Generate race predictions"""
    try:
        predictions = ml_model.get_race_predictions(
            drivers=request.drivers,
            circuit=request.circuit,
            weather=request.weather
        )
        
        # Update predictions file
        data = await load_json_file(PREDICTIONS_FILE)
        data["last_updated"] = datetime.now().isoformat()
        data["latest_prediction"] = {
            "circuit": request.circuit,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }
        await save_json_file(PREDICTIONS_FILE, data)
        
        return {"predictions": predictions, "success": True}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/api/analytics")
async def get_analytics():
    """Get analytics data"""
    # Generate sample analytics data
    analytics = {
        "model_performance": {
            "accuracy_trend": [
                {"race": "Race 1", "accuracy": 85.2},
                {"race": "Race 2", "accuracy": 87.1},
                {"race": "Race 3", "accuracy": 86.8},
                {"race": "Race 4", "accuracy": 89.3},
                {"race": "Race 5", "accuracy": 87.3}
            ],
            "prediction_confidence": [
                {"driver": "Max Verstappen", "avg_confidence": 94.2},
                {"driver": "Lewis Hamilton", "avg_confidence": 87.5},
                {"driver": "Charles Leclerc", "avg_confidence": 82.1},
                {"driver": "Lando Norris", "avg_confidence": 78.9}
            ]
        },
        "driver_performance": {
            "consistency_scores": [
                {"driver": "Max Verstappen", "score": 9.2},
                {"driver": "Lewis Hamilton", "score": 8.7},
                {"driver": "Charles Leclerc", "score": 7.9},
                {"driver": "Lando Norris", "score": 8.1}
            ]
        }
    }
    return analytics

@app.post("/api/retrain-model")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the ML model"""
    background_tasks.add_task(train_model_background)
    return {"message": "Model retraining started", "status": "in_progress"}

# Professional Pipeline Endpoints
@app.get("/api/professional/status")
async def get_professional_pipeline_status():
    """Get professional pipeline status"""
    if not PROFESSIONAL_PIPELINE_AVAILABLE:
        return {"error": "Professional pipeline not available", "basic_mode": True}
    
    return {
        "pipeline_status": pipeline_status,
        "timestamp": datetime.now().isoformat(),
        "professional_features_available": True
    }

class DataAcquisitionRequest(BaseModel):
    years: List[int]
    force_refresh: bool = False

@app.post("/api/professional/data/acquire")
async def acquire_professional_data(request: DataAcquisitionRequest, background_tasks: BackgroundTasks):
    """Acquire F1 data using professional pipeline"""
    if not PROFESSIONAL_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Professional pipeline not available")
    
    try:
        pipeline_status["data_acquisition"]["status"] = "running"
        
        def run_acquisition():
            try:
                consolidated_data = data_acquisition.acquire_multi_year_data(
                    years=request.years,
                    force_refresh=request.force_refresh
                )
                pipeline_status["data_acquisition"]["status"] = "completed"
                pipeline_status["data_acquisition"]["last_update"] = datetime.now().isoformat()
                return consolidated_data
            except Exception as e:
                pipeline_status["data_acquisition"]["status"] = "error"
                pipeline_status["data_acquisition"]["error"] = str(e)
        
        background_tasks.add_task(run_acquisition)
        
        return {
            "message": "Professional data acquisition started",
            "years": request.years,
            "status": "running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data acquisition failed: {str(e)}")

class StrategyRequest(BaseModel):
    driver: str
    circuit: str
    weather_forecast: Dict[str, Any]
    race_predictions: List[Dict[str, Any]]

@app.post("/api/professional/strategy/analyze")
async def analyze_professional_strategy(request: StrategyRequest):
    """Generate professional race strategy analysis"""
    if not PROFESSIONAL_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Professional pipeline not available")
    
    try:
        # Create mock historical data for demonstration
        import pandas as pd
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

@app.get("/api/professional/strategy/pit-optimization/{circuit}")
async def get_professional_pit_optimization(circuit: str, track_temp: float = 35, rain_prob: float = 0):
    """Get professional pit stop optimization"""
    if not PROFESSIONAL_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Professional pipeline not available")
    
    try:
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

class ProfessionalPredictionRequest(BaseModel):
    race_data: Dict[str, Any]
    weather: Dict[str, Any]
    use_enhanced_models: bool = True

@app.post("/api/professional/predictions")
async def get_professional_predictions(request: ProfessionalPredictionRequest):
    """Get predictions using professional ML pipeline"""
    if not PROFESSIONAL_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Professional pipeline not available")
    
    try:
        if request.use_enhanced_models:
            prediction_result = enhanced_models.predict_race_outcome(
                race_data=request.race_data,
                weather_conditions=request.weather
            )
        else:
            prediction_result = ml_model.predict_race_positions(
                race_data=request.race_data,
                weather=request.weather
            )
        
        return {
            "predictions": prediction_result,
            "model_type": "enhanced" if request.use_enhanced_models else "standard",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Professional prediction failed: {str(e)}")

@app.post("/api/professional/continuous/update")
async def trigger_professional_update(background_tasks: BackgroundTasks):
    """Trigger professional continuous update process"""
    if not PROFESSIONAL_PIPELINE_AVAILABLE:
        raise HTTPException(status_code=503, detail="Professional pipeline not available")
    
    try:
        pipeline_status["continuous_update"]["status"] = "running"
        
        def run_continuous_update():
            try:
                update_results = continuous_update.run_update_cycle()
                pipeline_status["continuous_update"]["status"] = "completed"
                pipeline_status["continuous_update"]["last_update"] = datetime.now().isoformat()
                return update_results
            except Exception as e:
                pipeline_status["continuous_update"]["status"] = "error"
                pipeline_status["continuous_update"]["error"] = str(e)
        
        background_tasks.add_task(run_continuous_update)
        
        return {
            "message": "Professional continuous update started",
            "status": "running"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Continuous update failed: {str(e)}")

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": len(ml_model.models) > 0,
        "professional_pipeline_available": PROFESSIONAL_PIPELINE_AVAILABLE
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
