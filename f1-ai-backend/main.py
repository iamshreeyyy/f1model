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

# Initialize FastAPI app
app = FastAPI(title="F1 AI Prediction API", version="1.0.0")

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

# Initialize ML model
ml_model = F1PredictionModel()

# Data storage paths
DATA_DIR = "data"
PREDICTIONS_FILE = f"{DATA_DIR}/predictions.json"
RACE_RESULTS_FILE = f"{DATA_DIR}/race_results.json"
DRIVER_STATS_FILE = f"{DATA_DIR}/driver_stats.json"
CHAMPIONSHIPS_FILE = f"{DATA_DIR}/championships.json"

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Pydantic models
class RacePredictionRequest(BaseModel):
    drivers: List[str]
    circuit: str
    weather: Optional[Dict[str, float]] = None

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

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": len(ml_model.models) > 0
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
