"""
F1 Advanced ML Training Pipeline
Implements professional ML pipeline with cross-validation, hyperparameter tuning, and model selection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, f1_score, roc_auc_score, log_loss
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import pickle
import joblib
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class F1MLPipeline:
    def __init__(self, models_dir: str = "data/models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = []
        self.performance_metrics = {}
        
        # Model configurations optimized for F1-score
        self.model_configs = {
            'xgboost': {
                'model': xgb.XGBClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 4, 6],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0],
                    'colsample_bytree': [0.8, 0.9, 1.0],
                    'eval_metric': ['logloss'],
                    'objective': ['binary:logistic']
                },
                'scoring': 'f1'
            },
            'random_forest': {
                'model': RandomForestClassifier,
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'class_weight': ['balanced', 'balanced_subsample']
                },
                'scoring': 'f1'
            },
            'gradient_boosting': {
                'model': GradientBoostingClassifier,
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 6],
                    'subsample': [0.8, 0.9, 1.0]
                },
                'scoring': 'f1'
            },
            'logistic_regression': {
                'model': LogisticRegression,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear', 'saga'],
                    'class_weight': ['balanced']
                },
                'scoring': 'f1'
            },
            'svm': {
                'model': SVC,
                'params': {
                    'C': [0.1, 1.0, 10.0],
                    'kernel': ['rbf', 'poly'],
                    'degree': [2, 3],
                    'gamma': ['scale', 'auto'],
                    'class_weight': ['balanced'],
                    'probability': [True]
                },
                'scoring': 'f1'
            }
        }
    
    def prepare_data(self, df: pd.DataFrame, feature_columns: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for training with proper time-series split
        """
        logger.info("Preparing data for training...")
        
        # Remove rows with missing target values
        df_clean = df.dropna(subset=['target_podium', 'target_points', 'target_win', 'target_dnf'])
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Split by season (train on earlier seasons, test on recent)
        train_seasons = df_clean['season'] <= 2023
        test_seasons = df_clean['season'] >= 2024
        
        train_df = df_clean[train_seasons].copy()
        test_df = df_clean[test_seasons].copy()
        
        logger.info(f"Training data: {len(train_df)} samples from seasons {train_df['season'].min()}-{train_df['season'].max()}")
        logger.info(f"Test data: {len(test_df)} samples from seasons {test_df['season'].min()}-{test_df['season'].max()}")
        
        return train_df, test_df
    
    def preprocess_features(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Preprocess features with scaling and encoding
        """
        logger.info("Preprocessing features...")
        
        # Select and prepare features
        X_train = train_df[self.feature_columns].copy()
        X_test = test_df[self.feature_columns].copy()
        
        # Handle missing values
        X_train = X_train.fillna(X_train.median())
        X_test = X_test.fillna(X_train.median())  # Use training median for test
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Store scaler
        self.scalers['feature_scaler'] = scaler
        
        return X_train_scaled, X_test_scaled
    
    def handle_class_imbalance(self, X_train: np.ndarray, y_train: np.ndarray, target: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using SMOTE or class weights
        """
        logger.info(f"Handling class imbalance for {target}...")
        
        # Check class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        
        logger.info(f"Class distribution for {target}: {class_distribution}")
        
        # Apply SMOTE for severely imbalanced classes
        minority_ratio = min(counts) / max(counts)
        
        if minority_ratio < 0.1:  # Less than 10% minority class
            smote = SMOTE(random_state=42, sampling_strategy='minority')
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
            logger.info(f"Applied SMOTE: {len(X_train)} -> {len(X_resampled)} samples")
            return X_resampled, y_resampled
        
        return X_train, y_train
    
    def train_model_for_target(self, X_train: np.ndarray, y_train: np.ndarray, 
                              X_test: np.ndarray, y_test: np.ndarray, 
                              target: str) -> Dict[str, Any]:
        """
        Train all models for a specific target with hyperparameter tuning
        """
        logger.info(f"Training models for target: {target}")
        
        # Handle class imbalance
        X_train_balanced, y_train_balanced = self.handle_class_imbalance(X_train, y_train, target)
        
        target_models = {}
        target_performance = {}
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        for model_name, config in self.model_configs.items():
            logger.info(f"Training {model_name} for {target}...")
            
            try:
                # Create model instance
                model_class = config['model']
                
                # Grid search with time series CV
                grid_search = GridSearchCV(
                    estimator=model_class(random_state=42),
                    param_grid=config['params'],
                    cv=tscv,
                    scoring=config['scoring'],
                    n_jobs=-1,
                    verbose=0
                )
                
                # Fit grid search
                grid_search.fit(X_train_balanced, y_train_balanced)
                
                # Get best model
                best_model = grid_search.best_estimator_
                
                # Evaluate on test set
                y_pred = best_model.predict(X_test)
                y_pred_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else y_pred
                
                # Calculate metrics
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Store results
                target_models[model_name] = best_model
                target_performance[model_name] = {
                    'f1_score': f1,
                    'auc_score': auc,
                    'best_params': grid_search.best_params_,
                    'cv_score': grid_search.best_score_
                }
                
                logger.info(f"{model_name} - F1: {f1:.3f}, AUC: {auc:.3f}")
                
            except Exception as e:
                logger.error(f"Error training {model_name} for {target}: {e}")
                continue
        
        return target_models, target_performance
    
    def train_all_targets(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models for all prediction targets
        """
        logger.info("Starting training for all targets...")
        
        # Preprocess features
        X_train, X_test = self.preprocess_features(train_df, test_df)
        
        # Define targets
        targets = ['target_podium', 'target_points', 'target_win', 'target_dnf']
        
        all_results = {}
        
        for target in targets:
            logger.info(f"\n{'='*50}")
            logger.info(f"TRAINING FOR TARGET: {target}")
            logger.info(f"{'='*50}")
            
            # Get target values
            y_train = train_df[target].values
            y_test = test_df[target].values
            
            # Train models for this target
            target_models, target_performance = self.train_model_for_target(
                X_train, y_train, X_test, y_test, target
            )
            
            # Store results
            self.models[target] = target_models
            self.performance_metrics[target] = target_performance
            
            all_results[target] = {
                'models': target_models,
                'performance': target_performance
            }
        
        return all_results
    
    def create_ensemble_predictions(self, X_test: np.ndarray, target: str) -> np.ndarray:
        """
        Create ensemble predictions using voting
        """
        if target not in self.models:
            raise ValueError(f"No models trained for target: {target}")
        
        predictions = []
        weights = []
        
        for model_name, model in self.models[target].items():
            pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_test)
            predictions.append(pred_proba)
            
            # Weight by F1 score
            f1_weight = self.performance_metrics[target][model_name]['f1_score']
            weights.append(f1_weight)
        
        # Weighted average
        weighted_predictions = np.average(predictions, axis=0, weights=weights)
        
        return weighted_predictions
    
    def save_pipeline(self, filepath: str = None):
        """
        Save the entire pipeline
        """
        if filepath is None:
            filepath = f"{self.models_dir}/f1_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        
        pipeline_data = {
            'models': self.models,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'performance_metrics': self.performance_metrics
        }
        
        joblib.dump(pipeline_data, filepath)
        logger.info(f"Pipeline saved to {filepath}")
        
        return filepath
    
    def load_pipeline(self, filepath: str):
        """
        Load a saved pipeline
        """
        pipeline_data = joblib.load(filepath)
        
        self.models = pipeline_data['models']
        self.scalers = pipeline_data['scalers']
        self.encoders = pipeline_data['encoders']
        self.feature_columns = pipeline_data['feature_columns']
        self.performance_metrics = pipeline_data['performance_metrics']
        
        logger.info(f"Pipeline loaded from {filepath}")
    
    def generate_model_report(self) -> str:
        """
        Generate comprehensive model performance report
        """
        report = []
        report.append("="*60)
        report.append("F1 ML PIPELINE PERFORMANCE REPORT")
        report.append("="*60)
        
        for target, models_perf in self.performance_metrics.items():
            report.append(f"\nTARGET: {target}")
            report.append("-" * 30)
            
            for model_name, metrics in models_perf.items():
                report.append(f"\n{model_name.upper()}:")
                report.append(f"  F1 Score: {metrics['f1_score']:.3f}")
                report.append(f"  AUC Score: {metrics['auc_score']:.3f}")
                report.append(f"  CV Score: {metrics['cv_score']:.3f}")
                report.append(f"  Best Params: {metrics['best_params']}")
        
        return "\n".join(report)

# Example usage and testing
if __name__ == "__main__":
    # Load engineered features
    df = pd.read_csv("data/f1_engineered_features.csv")
    
    # Initialize ML pipeline
    ml_pipeline = F1MLPipeline()
    
    # Get feature columns (exclude metadata and targets)
    feature_columns = [col for col in df.columns if col.startswith(('form_', 'circuit_', 'driver_', 'team_', 'career_')) 
                      or col in ['season', 'round', 'grid_position', 'grid_advantage']]
    
    # Prepare data
    train_df, test_df = ml_pipeline.prepare_data(df, feature_columns)
    
    # Train models
    results = ml_pipeline.train_all_targets(train_df, test_df)
    
    # Save pipeline
    pipeline_path = ml_pipeline.save_pipeline()
    
    # Generate report
    report = ml_pipeline.generate_model_report()
    print(report)
    
    # Save report
    with open("data/model_performance_report.txt", "w") as f:
        f.write(report)
