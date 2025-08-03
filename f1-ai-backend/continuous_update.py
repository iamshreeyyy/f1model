"""
F1 Continuous Update System
Automates data fetching, model retraining, and drift detection
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import json
import time
from pathlib import Path

logger = logging.getLogger(__name__)

class F1ContinuousUpdate:
    def __init__(self, data_dir: str = "data", models_dir: str = "data/models"):
        self.data_dir = data_dir
        self.models_dir = models_dir
        self.config_file = f"{data_dir}/update_config.json"
        self.metrics_history_file = f"{data_dir}/metrics_history.json"
        
        # Create directories
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        
        # Load or create configuration
        self.config = self._load_config()
        self.metrics_history = self._load_metrics_history()
        
        # Performance thresholds for drift detection
        self.drift_thresholds = {
            'f1_score_drop': 0.05,  # Trigger if F1 drops by 5%
            'auc_score_drop': 0.05,  # Trigger if AUC drops by 5%
            'prediction_accuracy_drop': 0.1  # Trigger if accuracy drops by 10%
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load update configuration"""
        default_config = {
            'last_update': None,
            'update_frequency_days': 7,  # Update weekly
            'auto_retrain': True,
            'retrain_threshold_races': 3,  # Retrain after 3 new races
            'monitoring_enabled': True,
            'data_sources': {
                'ergast_api': 'http://ergast.com/api/f1',
                'backup_enabled': True
            }
        }
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            return {**default_config, **config}
        except FileNotFoundError:
            self._save_config(default_config)
            return default_config
    
    def _save_config(self, config: Dict[str, Any]):
        """Save configuration"""
        with open(self.config_file, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def _load_metrics_history(self) -> List[Dict[str, Any]]:
        """Load historical metrics for drift detection"""
        try:
            with open(self.metrics_history_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return []
    
    def _save_metrics_history(self, metrics_history: List[Dict[str, Any]]):
        """Save metrics history"""
        with open(self.metrics_history_file, 'w') as f:
            json.dump(metrics_history, f, indent=2, default=str)
    
    def check_for_new_data(self) -> Dict[str, Any]:
        """
        Check if new race data is available
        """
        logger.info("Checking for new race data...")
        
        current_year = datetime.now().year
        last_update = self.config.get('last_update')
        
        if last_update:
            last_update_date = datetime.fromisoformat(last_update)
            days_since_update = (datetime.now() - last_update_date).days
        else:
            days_since_update = 999  # Force update if never updated
        
        # Check if it's time for an update
        update_needed = days_since_update >= self.config['update_frequency_days']
        
        # Check for new races (simplified - in production would query API)
        new_races_available = self._check_recent_races()
        
        return {
            'update_needed': update_needed,
            'days_since_update': days_since_update,
            'new_races_available': new_races_available,
            'last_update': last_update
        }
    
    def _check_recent_races(self) -> int:
        """
        Check how many recent races we don't have data for
        """
        # This is a simplified version - in production would query Ergast API
        try:
            # Load current data
            current_data_file = f"{self.data_dir}/all_f1_results.csv"
            if os.path.exists(current_data_file):
                df = pd.read_csv(current_data_file)
                latest_race_date = pd.to_datetime(df['date']).max()
                
                # Estimate races since last data (roughly 1 race every 2-3 weeks)
                days_since_latest = (datetime.now() - latest_race_date).days
                estimated_new_races = max(0, days_since_latest // 14)  # Approximately bi-weekly races
                
                return estimated_new_races
            else:
                return 999  # No data file exists, need full update
        except Exception as e:
            logger.error(f"Error checking recent races: {e}")
            return 0
    
    def fetch_incremental_data(self) -> pd.DataFrame:
        """
        Fetch only new race data since last update
        """
        logger.info("Fetching incremental race data...")
        
        # In production, this would:
        # 1. Query Ergast API for races since last update
        # 2. Append new data to existing dataset
        # 3. Handle race results, qualifying, practice sessions
        
        # For now, simulate new data
        new_data = self._simulate_new_race_data()
        
        if not new_data.empty:
            # Append to existing data
            existing_file = f"{self.data_dir}/all_f1_results.csv"
            if os.path.exists(existing_file):
                existing_data = pd.read_csv(existing_file)
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
            else:
                combined_data = new_data
            
            # Save updated dataset
            combined_data.to_csv(existing_file, index=False)
            logger.info(f"Added {len(new_data)} new race records")
            
            return new_data
        
        return pd.DataFrame()
    
    def _simulate_new_race_data(self) -> pd.DataFrame:
        """
        Simulate new race data for testing
        """
        # This is just for simulation - replace with actual API calls
        current_year = datetime.now().year
        
        # Create sample new race data
        drivers = ['Max Verstappen', 'Lando Norris', 'Charles Leclerc', 'Carlos Sainz', 'George Russell']
        
        new_race_data = []
        for i, driver in enumerate(drivers):
            new_race_data.append({
                'season': current_year,
                'round': 1,  # Simplified
                'grand_prix': 'Bahrain Grand Prix',
                'date': datetime.now().strftime('%Y-%m-%d'),
                'driver': driver,
                'constructor': 'Red Bull' if 'Verstappen' in driver else 'McLaren',
                'grid_position': i + 1,
                'finishing_position': i + 1,
                'laps_completed': 57,
                'points_earned': [25, 18, 15, 12, 10][i],
                'status': 'Finished'
            })
        
        return pd.DataFrame(new_race_data)
    
    def detect_model_drift(self, new_predictions: Dict[str, np.ndarray], 
                          actual_results: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Detect if model performance has degraded (drift detection)
        """
        logger.info("Checking for model drift...")
        
        drift_detected = False
        drift_details = {}
        
        # Calculate current performance metrics
        current_metrics = self._calculate_performance_metrics(new_predictions, actual_results)
        
        # Compare with historical performance
        if self.metrics_history:
            # Get recent historical average
            recent_metrics = self.metrics_history[-5:]  # Last 5 evaluations
            historical_avg = self._calculate_historical_average(recent_metrics)
            
            # Check for significant drops
            for metric_name, current_value in current_metrics.items():
                if metric_name in historical_avg:
                    historical_value = historical_avg[metric_name]
                    drop = historical_value - current_value
                    relative_drop = drop / historical_value if historical_value > 0 else 0
                    
                    threshold_key = f"{metric_name}_drop"
                    if threshold_key in self.drift_thresholds:
                        if relative_drop > self.drift_thresholds[threshold_key]:
                            drift_detected = True
                            drift_details[metric_name] = {
                                'current': current_value,
                                'historical_avg': historical_value,
                                'drop': drop,
                                'relative_drop': relative_drop
                            }
        
        # Add current metrics to history
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics
        })
        
        # Keep only last 20 evaluations
        self.metrics_history = self.metrics_history[-20:]
        self._save_metrics_history(self.metrics_history)
        
        return {
            'drift_detected': drift_detected,
            'drift_details': drift_details,
            'current_metrics': current_metrics
        }
    
    def _calculate_performance_metrics(self, predictions: Dict[str, np.ndarray], 
                                     actuals: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate performance metrics for drift detection
        """
        metrics = {}
        
        for target in predictions.keys():
            if target in actuals:
                pred = predictions[target]
                actual = actuals[target]
                
                # Simple accuracy for binary predictions
                if len(np.unique(actual)) == 2:  # Binary classification
                    pred_binary = (pred > 0.5).astype(int)
                    accuracy = np.mean(pred_binary == actual)
                    metrics[f"{target}_accuracy"] = accuracy
                    
                    # F1 score approximation
                    tp = np.sum((pred_binary == 1) & (actual == 1))
                    fp = np.sum((pred_binary == 1) & (actual == 0))
                    fn = np.sum((pred_binary == 0) & (actual == 1))
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    metrics[f"{target}_f1_score"] = f1
        
        return metrics
    
    def _calculate_historical_average(self, recent_metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate average of recent historical metrics
        """
        if not recent_metrics:
            return {}
        
        all_metrics = {}
        for entry in recent_metrics:
            for metric_name, value in entry['metrics'].items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)
        
        # Calculate averages
        avg_metrics = {}
        for metric_name, values in all_metrics.items():
            avg_metrics[metric_name] = np.mean(values)
        
        return avg_metrics
    
    def trigger_retraining(self, reason: str = "Scheduled update") -> Dict[str, Any]:
        """
        Trigger model retraining
        """
        logger.info(f"Triggering model retraining. Reason: {reason}")
        
        try:
            # This would normally call the ML pipeline
            # For now, just simulate the process
            
            retrain_start = datetime.now()
            
            # Simulate retraining process
            logger.info("Loading updated data...")
            time.sleep(1)  # Simulate data loading
            
            logger.info("Re-engineering features...")
            time.sleep(2)  # Simulate feature engineering
            
            logger.info("Training models...")
            time.sleep(5)  # Simulate model training
            
            logger.info("Validating models...")
            time.sleep(1)  # Simulate validation
            
            retrain_end = datetime.now()
            duration = (retrain_end - retrain_start).total_seconds()
            
            # Update configuration
            self.config['last_update'] = datetime.now().isoformat()
            self._save_config(self.config)
            
            return {
                'success': True,
                'duration_seconds': duration,
                'timestamp': retrain_end.isoformat(),
                'reason': reason
            }
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def run_update_cycle(self) -> Dict[str, Any]:
        """
        Run complete update cycle: check data, detect drift, retrain if needed
        """
        logger.info("Starting F1 continuous update cycle...")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'data_check': None,
            'new_data_fetched': False,
            'drift_check': None,
            'retrain_triggered': False,
            'retrain_result': None
        }
        
        # 1. Check for new data
        data_check = self.check_for_new_data()
        results['data_check'] = data_check
        
        # 2. Fetch new data if available
        if data_check['new_races_available'] > 0:
            new_data = self.fetch_incremental_data()
            results['new_data_fetched'] = not new_data.empty
        
        # 3. Check for model drift (simplified - would need actual predictions)
        # In production, this would use recent race predictions vs actual results
        
        # 4. Determine if retraining is needed
        retrain_needed = (
            data_check['update_needed'] or 
            data_check['new_races_available'] >= self.config['retrain_threshold_races']
        )
        
        if retrain_needed and self.config['auto_retrain']:
            retrain_result = self.trigger_retraining("New data available")
            results['retrain_triggered'] = True
            results['retrain_result'] = retrain_result
        
        logger.info("Update cycle complete")
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status and health metrics
        """
        return {
            'last_update': self.config.get('last_update'),
            'auto_retrain_enabled': self.config.get('auto_retrain', False),
            'monitoring_enabled': self.config.get('monitoring_enabled', False),
            'metrics_history_length': len(self.metrics_history),
            'drift_thresholds': self.drift_thresholds,
            'data_dir': self.data_dir,
            'models_dir': self.models_dir
        }
    
    def schedule_updates(self, frequency_days: int = 7):
        """
        Update the scheduled update frequency
        """
        self.config['update_frequency_days'] = frequency_days
        self._save_config(self.config)
        logger.info(f"Update frequency set to {frequency_days} days")

# Example usage
if __name__ == "__main__":
    # Initialize continuous update system
    updater = F1ContinuousUpdate()
    
    # Run update cycle
    results = updater.run_update_cycle()
    
    print("Update Cycle Results:")
    print(json.dumps(results, indent=2, default=str))
    
    # Get system status
    status = updater.get_system_status()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2, default=str))
