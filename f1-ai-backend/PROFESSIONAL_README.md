# F1 Professional ML Pipeline 🏎️

A comprehensive, enterprise-grade machine learning pipeline for Formula 1 race prediction and strategic analysis. This system provides real-time predictions, strategic insights, and continuous model improvement capabilities.

## 🚀 Features

### Core ML Pipeline
- **Automated Data Acquisition**: Real-time data fetching from Ergast Motor Racing API (2011-present)
- **Advanced Feature Engineering**: 50+ engineered features including rolling averages, circuit characteristics, and driver form
- **Professional Model Training**: Ensemble methods with cross-validation and hyperparameter optimization
- **Continuous Updates**: Automated model retraining with drift detection

### Strategic Decision Support
- **Race Strategy Analysis**: Comprehensive pre-race strategic recommendations
- **Pit Stop Optimization**: Circuit-specific pit window analysis with weather integration
- **Driver Performance Analysis**: Historical performance trends and circuit-specific insights
- **Real-time Predictions**: Enhanced ML models with F1-score optimization

### API Capabilities
- **RESTful API**: Full FastAPI implementation with automatic documentation
- **Real-time Processing**: Asynchronous background tasks for long-running operations
- **Health Monitoring**: Comprehensive pipeline health checks and metrics
- **CORS Support**: Frontend integration ready

## 📁 Project Structure

```
f1-ai-backend/
├── main.py                     # Main FastAPI application with professional endpoints
├── professional_api.py         # Dedicated professional pipeline API
├── model.py                    # Base F1 prediction model
├── enhanced_f1_models.py       # F1-score optimized classification models
├── data_acquisition.py         # Professional data fetching from Ergast API
├── feature_engineering.py      # Advanced feature engineering pipeline
├── ml_pipeline.py              # Complete ML training pipeline
├── continuous_update.py        # Automated update and monitoring system
├── decision_support.py         # Strategic analysis and recommendations
├── start_professional.sh       # Professional pipeline startup script
├── requirements.txt            # All necessary dependencies
└── data/
    ├── cache/                  # API response caching
    ├── models/                 # Trained model storage
    ├── f1_consolidated_data.csv    # Raw consolidated data
    ├── f1_engineered_features.csv # Processed features
    ├── predictions.json        # Prediction history
    ├── race_results.json       # Race results database
    └── driver_stats.json       # Driver statistics
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.9+
- Virtual environment support
- 2GB+ available disk space
- Internet connection for data acquisition

### Quick Start

1. **Clone and Navigate**
   ```bash
   cd "f1-ai-backend"
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv f1_env
   source f1_env/bin/activate  # Linux/Mac
   # or
   f1_env\Scripts\activate     # Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Professional Pipeline**
   ```bash
   ./start_professional.sh
   ```

The API will be available at `http://localhost:8000` with automatic documentation at `http://localhost:8000/docs`.

## 🔧 API Endpoints

### Core Endpoints
- `GET /api/health` - Health check and system status
- `GET /api/professional/status` - Professional pipeline status
- `POST /api/predictions` - Basic race predictions
- `POST /api/enhanced-predictions` - Enhanced ML predictions

### Professional Pipeline Endpoints
- `POST /api/professional/data/acquire` - Acquire F1 data for specified years
- `POST /api/professional/predictions` - Enhanced predictions with professional models
- `POST /api/professional/strategy/analyze` - Comprehensive race strategy analysis
- `GET /api/professional/strategy/pit-optimization/{circuit}` - Pit stop optimization
- `POST /api/professional/continuous/update` - Trigger continuous update process

### Data Management
- `GET /api/predictions` - Historical predictions
- `GET /api/race-results` - Race results database
- `GET /api/driver-stats` - Driver statistics
- `GET /api/championships` - Championship standings

## 🧠 ML Pipeline Components

### 1. Data Acquisition (`data_acquisition.py`)
```python
# Acquire data for multiple years
data_acquisition = F1DataAcquisition()
data = data_acquisition.acquire_multi_year_data(years=[2020, 2021, 2022, 2023])
```

**Features:**
- Ergast Motor Racing API integration
- Comprehensive data validation
- Automatic schema standardization
- Intelligent caching system

### 2. Feature Engineering (`feature_engineering.py`)
```python
# Engineer advanced features
feature_engineering = F1FeatureEngineering()
features = feature_engineering.create_comprehensive_features("data/raw_data.csv")
```

**Generated Features:**
- Rolling performance averages (3, 5, 10 races)
- Circuit-specific characteristics
- Driver form and momentum indicators
- Team performance metrics
- Weather impact factors

### 3. ML Pipeline (`ml_pipeline.py`)
```python
# Train ensemble models
ml_pipeline = F1MLPipeline()
results = ml_pipeline.train_ensemble_models("data/features.csv")
```

**Capabilities:**
- Time-series cross-validation
- Hyperparameter optimization (RandomizedSearchCV)
- Class imbalance handling (SMOTE)
- Ensemble predictions (Voting Classifier)
- Model performance tracking

### 4. Continuous Updates (`continuous_update.py`)
```python
# Monitor and update system
continuous_update = F1ContinuousUpdate()
update_results = continuous_update.run_update_cycle()
```

**Features:**
- Data drift detection
- Performance monitoring
- Automated retraining triggers
- Incremental data updates

### 5. Decision Support (`decision_support.py`)
```python
# Generate strategic insights
decision_support = F1DecisionSupport()
strategy = decision_support.generate_race_strategy_report(
    driver="Max Verstappen",
    circuit="Monaco",
    race_predictions=predictions,
    weather_forecast=weather,
    historical_data=data
)
```

**Strategic Analysis:**
- Circuit-specific performance analysis
- Qualifying strategy optimization
- Pit stop window predictions
- Risk factor identification

## 📊 Enhanced ML Models

The system includes F1-score optimized classification models:

### Model Types
1. **XGBoost Classifier** - Gradient boosting with advanced regularization
2. **Random Forest** - Ensemble decision trees with feature importance
3. **Gradient Boosting** - Sequential weak learner optimization
4. **Logistic Regression** - Linear classification with regularization
5. **Support Vector Machine** - High-dimensional classification

### Performance Metrics
- **F1-Score Optimization**: Primary metric for model selection
- **Precision/Recall Balance**: Optimized for racing context
- **Cross-Validation**: Time-series aware validation
- **Feature Importance**: Automated feature selection

## 🔄 Continuous Integration

### Automated Updates
The system automatically:
1. **Monitors Data Quality**: Detects data drift and anomalies
2. **Tracks Performance**: Model performance degradation detection
3. **Updates Models**: Retrains when performance thresholds are met
4. **Validates Changes**: Comprehensive testing before deployment

### Update Triggers
- Model performance below threshold (configurable)
- Data drift detection beyond limits
- Manual trigger via API
- Scheduled updates (weekly/monthly)

## 🎯 Strategic Decision Support

### Race Strategy Analysis
```json
{
  "driver": "Max Verstappen",
  "circuit": "Monaco",
  "historical_performance": {
    "avg_position": 2.3,
    "podium_rate": 0.8,
    "performance_trend": "improving"
  },
  "qualifying_strategy": {
    "qualifying_importance_score": 10,
    "recommendations": ["HIGH PRIORITY: Qualifying position crucial at Monaco"]
  },
  "pit_stop_strategy": {
    "optimal_window": [35, 45],
    "strategy": "defensive",
    "notes": "Protect track position, react to competitors"
  }
}
```

### Pit Stop Optimization
- Circuit-specific pit windows
- Weather condition adjustments
- Tire degradation modeling
- Safety car probability analysis

## 📈 Monitoring & Analytics

### Health Checks
- Component availability
- Data file integrity
- Model performance metrics
- Pipeline efficiency tracking

### Performance Metrics
- Prediction accuracy over time
- Model drift detection
- API response times
- Data quality scores

## 🚦 Usage Examples

### Basic Prediction
```bash
curl -X POST "http://localhost:8000/api/predictions" \
  -H "Content-Type: application/json" \
  -d '{
    "race_data": {
      "circuit": "silverstone",
      "drivers": ["Max Verstappen", "Lewis Hamilton"]
    },
    "weather": {"condition": "dry", "temperature": 25}
  }'
```

### Professional Strategy Analysis
```bash
curl -X POST "http://localhost:8000/api/professional/strategy/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "driver": "Max Verstappen",
    "circuit": "Monaco",
    "weather_forecast": {"rain_probability": 0.3, "temperature": 28},
    "race_predictions": [
      {"driver": "Max Verstappen", "predicted_position": 1}
    ]
  }'
```

### Data Acquisition
```bash
curl -X POST "http://localhost:8000/api/professional/data/acquire" \
  -H "Content-Type: application/json" \
  -d '{
    "years": [2023, 2024],
    "force_refresh": false
  }'
```

## 🔍 Troubleshooting

### Common Issues

1. **ImportError for professional components**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.9+)

2. **API connection errors**
   - Verify CORS configuration for your frontend port
   - Check firewall settings for port 8000

3. **Data acquisition failures**
   - Check internet connection
   - Verify Ergast API availability
   - Review rate limiting settings

4. **Model training issues**
   - Ensure sufficient disk space (2GB+)
   - Check data quality and completeness
   - Verify memory availability for large datasets

### Performance Optimization

1. **Caching Strategy**
   - Enable API response caching
   - Use incremental data updates
   - Implement model result caching

2. **Resource Management**
   - Monitor memory usage during training
   - Use background tasks for long operations
   - Implement request queuing for high load

## 🔒 Security Considerations

- API rate limiting implementation
- Input validation and sanitization
- Secure file handling for data storage
- Environment variable configuration for sensitive data

## 🚀 Production Deployment

### Recommended Setup
1. **Containerization**: Docker deployment with multi-stage builds
2. **Load Balancing**: Multiple API instances behind load balancer
3. **Database**: Production database for data persistence
4. **Monitoring**: Comprehensive logging and metrics collection
5. **Backup**: Automated model and data backups

### Scaling Considerations
- Horizontal scaling with stateless API design
- Model serving optimization
- Data pipeline parallelization
- Caching layer implementation

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation at `/docs`
3. Monitor system logs for error details
4. Check pipeline status at `/api/professional/status`

---

**Built with ❤️ for Formula 1 enthusiasts and data scientists**
