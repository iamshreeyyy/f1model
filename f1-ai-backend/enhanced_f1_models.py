import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import xgboost as xgb
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class EnhancedF1PredictionModels:
    """
    Enhanced F1 Race Prediction Models optimized for performance.
    
    This class implements multiple ML models specifically optimized for F1 race predictions:
    - XGBoost with F1-score optimization
    - Random Forest with class balancing
    - Gradient Boosting with custom objectives
    - Logistic Regression with threshold optimization
    - SVM with kernel optimization
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.best_thresholds = {}
        self.feature_importance = {}
        
    def create_classification_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create classification targets for F1 predictions:
        - Top 3 finish (podium)
        - Points finish (top 10)
        - DNF prediction
        - Pole position
        """
        data = df.copy()
        
        # Podium prediction (1-3 positions)
        data['podium'] = (data['Position'] <= 3).astype(int)
        
        # Points finish (1-10 positions)
        data['points_finish'] = (data['Position'] <= 10).astype(int)
        
        # DNF prediction (position > 20)
        data['dnf'] = (data['Position'] > 20).astype(int)
        
        # Winner prediction
        data['winner'] = (data['Position'] == 1).astype(int)
        
        # Grid position advantage (qualify better than finish)
        data['grid_advantage'] = (data['GridPosition'] < data['Position']).astype(int)
        
        return data
        
    def train_xgboost_f1_optimized(self, X: pd.DataFrame, y: pd.Series, 
                                  target_name: str) -> xgb.XGBClassifier:
        """
        Train XGBoost with F1-score optimization
        """
        print(f"🚀 Training XGBoost for {target_name} with F1 optimization...")
        
        # XGBoost parameters optimized for F1 score
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'aucpr'],  # AUC-PR is related to F1
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1, 0.2],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'scale_pos_weight': [1, len(y[y==0])/len(y[y==1])]  # Handle class imbalance
        }
        
        xgb_model = xgb.XGBClassifier(random_state=42)
        
        # Grid search with F1 scoring
        grid_search = GridSearchCV(
            xgb_model, 
            xgb_params, 
            scoring='f1',
            cv=5, 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        f1_scores = cross_val_score(best_model, X, y, scoring='f1', cv=5)
        
        print(f"✅ XGBoost {target_name} - Best F1 Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
        print(f"📊 Best Parameters: {grid_search.best_params_}")
        
        return best_model
    
    def train_random_forest_balanced(self, X: pd.DataFrame, y: pd.Series, 
                                   target_name: str) -> RandomForestClassifier:
        """
        Train Random Forest with class balancing for F1 optimization
        """
        print(f"🌲 Training Random Forest for {target_name} with class balancing...")
        
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample'],  # Handle imbalanced data
            'bootstrap': [True, False]
        }
        
        rf_model = RandomForestClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            rf_model, 
            rf_params, 
            scoring='f1',
            cv=5, 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        f1_scores = cross_val_score(best_model, X, y, scoring='f1', cv=5)
        
        print(f"✅ Random Forest {target_name} - Best F1 Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
        
        # Store feature importance
        self.feature_importance[f'rf_{target_name}'] = dict(zip(X.columns, best_model.feature_importances_))
        
        return best_model
    
    def train_gradient_boosting_custom(self, X: pd.DataFrame, y: pd.Series, 
                                     target_name: str) -> GradientBoostingClassifier:
        """
        Train Gradient Boosting with custom F1 optimization
        """
        print(f"📈 Training Gradient Boosting for {target_name} with custom optimization...")
        
        gb_params = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7],
            'subsample': [0.8, 0.9, 1.0],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        gb_model = GradientBoostingClassifier(random_state=42)
        
        grid_search = GridSearchCV(
            gb_model, 
            gb_params, 
            scoring='f1',
            cv=5, 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        best_model = grid_search.best_estimator_
        f1_scores = cross_val_score(best_model, X, y, scoring='f1', cv=5)
        
        print(f"✅ Gradient Boosting {target_name} - Best F1 Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
        
        return best_model
    
    def train_logistic_regression_optimized(self, X: pd.DataFrame, y: pd.Series, 
                                          target_name: str) -> Tuple[LogisticRegression, float]:
        """
        Train Logistic Regression with threshold optimization for F1
        """
        print(f"📊 Training Logistic Regression for {target_name} with threshold optimization...")
        
        # Scale features for logistic regression
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[f'lr_{target_name}'] = scaler
        
        lr_params = {
            'C': [0.01, 0.1, 1.0, 10.0, 100.0],
            'class_weight': [None, 'balanced'],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
        
        lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
        grid_search = GridSearchCV(
            lr_model, 
            lr_params, 
            scoring='f1',
            cv=5, 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        best_model = grid_search.best_estimator_
        
        # Optimize threshold for F1 score
        y_proba = best_model.predict_proba(X_scaled)[:, 1]
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred_thresh = (y_proba >= threshold).astype(int)
            f1 = f1_score(y, y_pred_thresh)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_thresholds[f'lr_{target_name}'] = best_threshold
        
        print(f"✅ Logistic Regression {target_name} - Best F1 Score: {best_f1:.3f}")
        print(f"🎯 Optimal Threshold: {best_threshold:.3f}")
        
        return best_model, best_threshold
    
    def train_svm_kernel_optimized(self, X: pd.DataFrame, y: pd.Series, 
                                 target_name: str) -> SVC:
        """
        Train SVM with kernel optimization for F1 score
        """
        print(f"🔧 Training SVM for {target_name} with kernel optimization...")
        
        # Scale features for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        self.scalers[f'svm_{target_name}'] = scaler
        
        svm_params = {
            'C': [0.1, 1.0, 10.0, 100.0],
            'kernel': ['linear', 'rbf', 'poly'],
            'class_weight': [None, 'balanced'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
        }
        
        svm_model = SVC(random_state=42, probability=True)
        
        grid_search = GridSearchCV(
            svm_model, 
            svm_params, 
            scoring='f1',
            cv=5, 
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_scaled, y)
        
        best_model = grid_search.best_estimator_
        f1_scores = cross_val_score(best_model, X_scaled, y, scoring='f1', cv=5)
        
        print(f"✅ SVM {target_name} - Best F1 Score: {f1_scores.mean():.3f} ± {f1_scores.std():.3f}")
        
        return best_model
    
    def train_ensemble_f1_optimized(self, X: pd.DataFrame, y: pd.Series, target_name: str):
        """
        Train ensemble of all models optimized for F1 score
        """
        print(f"\n🎯 Training Ensemble Models for {target_name.upper()} Prediction")
        print("=" * 60)
        
        # Train all models
        models = {}
        
        # XGBoost
        models['xgb'] = self.train_xgboost_f1_optimized(X, y, target_name)
        
        # Random Forest
        models['rf'] = self.train_random_forest_balanced(X, y, target_name)
        
        # Gradient Boosting
        models['gb'] = self.train_gradient_boosting_custom(X, y, target_name)
        
        # Logistic Regression
        models['lr'], threshold = self.train_logistic_regression_optimized(X, y, target_name)
        
        # SVM
        models['svm'] = self.train_svm_kernel_optimized(X, y, target_name)
        
        self.models[target_name] = models
        
        print(f"\n✅ Ensemble training completed for {target_name}!")
        return models
    
    def predict_ensemble(self, X: pd.DataFrame, target_name: str) -> np.ndarray:
        """
        Make ensemble predictions using all trained models
        """
        if target_name not in self.models:
            raise ValueError(f"Models for {target_name} not trained yet!")
        
        models = self.models[target_name]
        predictions = []
        
        for model_name, model in models.items():
            if model_name in ['lr', 'svm'] and f'{model_name}_{target_name}' in self.scalers:
                # Scale features for models that need it
                X_scaled = self.scalers[f'{model_name}_{target_name}'].transform(X)
                if model_name == 'lr' and f'lr_{target_name}' in self.best_thresholds:
                    # Use optimized threshold for logistic regression
                    y_proba = model.predict_proba(X_scaled)[:, 1]
                    threshold = self.best_thresholds[f'lr_{target_name}']
                    pred = (y_proba >= threshold).astype(int)
                else:
                    pred = model.predict(X_scaled)
            else:
                pred = model.predict(X)
            
            predictions.append(pred)
        
        # Ensemble prediction (majority vote)
        ensemble_pred = np.round(np.mean(predictions, axis=0)).astype(int)
        return ensemble_pred
    
    def get_model_performance_report(self, X: pd.DataFrame, y: pd.Series, target_name: str):
        """
        Generate comprehensive performance report for all models
        """
        if target_name not in self.models:
            print(f"❌ Models for {target_name} not trained yet!")
            return
        
        print(f"\n📊 PERFORMANCE REPORT: {target_name.upper()}")
        print("=" * 50)
        
        models = self.models[target_name]
        
        for model_name, model in models.items():
            if model_name in ['lr', 'svm'] and f'{model_name}_{target_name}' in self.scalers:
                X_test = self.scalers[f'{model_name}_{target_name}'].transform(X)
            else:
                X_test = X
            
            y_pred = model.predict(X_test)
            f1 = f1_score(y, y_pred)
            
            print(f"\n{model_name.upper()} Model:")
            print(f"F1 Score: {f1:.3f}")
            print(f"Classification Report:")
            print(classification_report(y, y_pred))
        
        # Ensemble performance
        ensemble_pred = self.predict_ensemble(X, target_name)
        ensemble_f1 = f1_score(y, ensemble_pred)
        
        print(f"\n🏆 ENSEMBLE Model:")
        print(f"F1 Score: {ensemble_f1:.3f}")
        print(f"Classification Report:")
        print(classification_report(y, ensemble_pred))
        
        return ensemble_f1

def example_usage():
    """
    Example of how to use the enhanced F1 prediction models
    """
    # Initialize the enhanced models
    f1_models = EnhancedF1PredictionModels()
    
    # Create sample data (replace with your actual F1 data)
    np.random.seed(42)
    n_samples = 1000
    n_features = 16
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create sample classification targets
    y_podium = np.random.binomial(1, 0.15, n_samples)  # 15% podium rate
    y_points = np.random.binomial(1, 0.5, n_samples)   # 50% points rate
    y_winner = np.random.binomial(1, 0.05, n_samples)  # 5% win rate
    
    print("🏁 Starting Enhanced F1 Prediction Model Training...")
    
    # Train models for different prediction targets
    targets = {
        'podium': y_podium,
        'points_finish': y_points, 
        'winner': y_winner
    }
    
    for target_name, y in targets.items():
        f1_models.train_ensemble_f1_optimized(X, y, target_name)
        f1_models.get_model_performance_report(X, y, target_name)

if __name__ == "__main__":
    example_usage()
