"""
F1 Feature Engineering Module
Creates predictive features for race outcome prediction
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class F1FeatureEngineering:
    def __init__(self):
        # Circuit characteristics lookup table
        self.circuit_characteristics = {
            'monaco': {'length_km': 3.337, 'turns': 19, 'overtaking_difficulty': 9, 'avg_lap_time': 78.0},
            'silverstone': {'length_km': 5.891, 'turns': 18, 'overtaking_difficulty': 4, 'avg_lap_time': 91.0},
            'spa': {'length_km': 7.004, 'turns': 19, 'overtaking_difficulty': 3, 'avg_lap_time': 107.0},
            'monza': {'length_km': 5.793, 'turns': 11, 'overtaking_difficulty': 2, 'avg_lap_time': 81.0},
            'suzuka': {'length_km': 5.807, 'turns': 18, 'overtaking_difficulty': 6, 'avg_lap_time': 91.0},
            'interlagos': {'length_km': 4.309, 'turns': 15, 'overtaking_difficulty': 5, 'avg_lap_time': 74.0},
            'austin': {'length_km': 5.513, 'turns': 20, 'overtaking_difficulty': 4, 'avg_lap_time': 95.0},
            'bahrain': {'length_km': 5.412, 'turns': 15, 'overtaking_difficulty': 3, 'avg_lap_time': 93.0},
            'jeddah': {'length_km': 6.174, 'turns': 27, 'overtaking_difficulty': 5, 'avg_lap_time': 92.0},
            'imola': {'length_km': 4.909, 'turns': 19, 'overtaking_difficulty': 7, 'avg_lap_time': 81.0},
            'miami': {'length_km': 5.41, 'turns': 19, 'overtaking_difficulty': 4, 'avg_lap_time': 92.0},
            'barcelona': {'length_km': 4.675, 'turns': 16, 'overtaking_difficulty': 6, 'avg_lap_time': 79.0},
            'baku': {'length_km': 6.003, 'turns': 20, 'overtaking_difficulty': 3, 'avg_lap_time': 103.0},
            'canada': {'length_km': 4.361, 'turns': 14, 'overtaking_difficulty': 4, 'avg_lap_time': 73.0},
            'austria': {'length_km': 4.318, 'turns': 10, 'overtaking_difficulty': 3, 'avg_lap_time': 70.0},
            'hungary': {'length_km': 4.381, 'turns': 14, 'overtaking_difficulty': 8, 'avg_lap_time': 81.0},
            'zandvoort': {'length_km': 4.259, 'turns': 14, 'overtaking_difficulty': 7, 'avg_lap_time': 72.0},
            'singapore': {'length_km': 5.063, 'turns': 23, 'overtaking_difficulty': 8, 'avg_lap_time': 103.0},
            'qatar': {'length_km': 5.38, 'turns': 16, 'overtaking_difficulty': 4, 'avg_lap_time': 84.0},
            'mexico': {'length_km': 4.304, 'turns': 17, 'overtaking_difficulty': 5, 'avg_lap_time': 78.0},
            'vegas': {'length_km': 6.201, 'turns': 17, 'overtaking_difficulty': 3, 'avg_lap_time': 96.0},
            'abu_dhabi': {'length_km': 5.281, 'turns': 16, 'overtaking_difficulty': 4, 'avg_lap_time': 95.0}
        }
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main method to create all predictive features
        """
        logger.info("Starting feature engineering...")
        
        # Ensure data is sorted by date
        df = df.sort_values(['season', 'round', 'grid_position']).reset_index(drop=True)
        
        # Create base features
        df = self._create_driver_form_features(df)
        df = self._create_circuit_features(df)
        df = self._create_qualifying_features(df)
        df = self._create_team_features(df)
        df = self._create_historical_features(df)
        df = self._create_target_variables(df)
        
        logger.info(f"Feature engineering complete. Added {len([c for c in df.columns if c.endswith('_feature') or 'form_' in c or 'circuit_' in c])} features")
        
        return df
    
    def _create_driver_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create rolling form features for each driver
        """
        logger.info("Creating driver form features...")
        
        # Sort by driver and date
        df = df.sort_values(['driver', 'season', 'round'])
        
        # Rolling averages for last N races
        windows = [3, 5, 10]
        
        for window in windows:
            # Rolling average finishing position
            df[f'form_avg_position_{window}races'] = df.groupby('driver')['finishing_position'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Rolling podium rate
            df['is_podium'] = (df['finishing_position'] <= 3).astype(int)
            df[f'form_podium_rate_{window}races'] = df.groupby('driver')['is_podium'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Rolling points rate
            df['earned_points'] = (df['points_earned'] > 0).astype(int)
            df[f'form_points_rate_{window}races'] = df.groupby('driver')['earned_points'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
            
            # Rolling DNF rate
            df['is_dnf'] = (~df['finishing_position'].notna()).astype(int)
            df[f'form_dnf_rate_{window}races'] = df.groupby('driver')['is_dnf'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # Career statistics (up to current race)
        df['career_races'] = df.groupby('driver').cumcount()
        df['career_wins'] = df.groupby('driver')['finishing_position'].transform(
            lambda x: (x == 1).cumsum().shift(1)
        )
        df['career_podiums'] = df.groupby('driver')['is_podium'].transform(
            lambda x: x.cumsum().shift(1)
        )
        df['career_avg_position'] = df.groupby('driver')['finishing_position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        return df
    
    def _create_circuit_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add circuit-specific features
        """
        logger.info("Creating circuit features...")
        
        # Extract circuit name from circuit_id or grand_prix
        df['circuit_name'] = df['grand_prix'].str.lower().str.replace(' grand prix', '').str.replace(' ', '_')
        
        # Add circuit characteristics
        for characteristic in ['length_km', 'turns', 'overtaking_difficulty', 'avg_lap_time']:
            df[f'circuit_{characteristic}'] = df['circuit_name'].map(
                lambda x: self.circuit_characteristics.get(x, {}).get(characteristic, 
                    np.mean([v[characteristic] for v in self.circuit_characteristics.values()]))
            )
        
        # Driver performance at specific circuit (historical)
        df = df.sort_values(['driver', 'circuit_name', 'season', 'round'])
        df['driver_circuit_avg_position'] = df.groupby(['driver', 'circuit_name'])['finishing_position'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        df['driver_circuit_best_position'] = df.groupby(['driver', 'circuit_name'])['finishing_position'].transform(
            lambda x: x.expanding().min().shift(1)
        )
        df['driver_circuit_races'] = df.groupby(['driver', 'circuit_name']).cumcount()
        
        return df
    
    def _create_qualifying_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create qualifying-related features
        """
        logger.info("Creating qualifying features...")
        
        # Grid position advantage (lower is better)
        df['grid_advantage'] = 21 - df['grid_position']  # Invert so higher is better
        
        # Qualifying vs race performance
        df['grid_to_finish_change'] = df['grid_position'] - df['finishing_position']
        
        # Historical qualifying vs race correlation for driver
        df = df.sort_values(['driver', 'season', 'round'])
        df['driver_avg_grid_to_finish'] = df.groupby('driver')['grid_to_finish_change'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        return df
    
    def _create_team_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create team/constructor-related features
        """
        logger.info("Creating team features...")
        
        # Team form
        df = df.sort_values(['constructor', 'season', 'round'])
        
        # Team average position in recent races
        for window in [3, 5]:
            df[f'team_form_avg_position_{window}races'] = df.groupby('constructor')['finishing_position'].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean().shift(1)
            )
        
        # Team championship position (season-to-date)
        df['season_round'] = df['season'] * 100 + df['round']
        
        # Team reliability (DNF rate)
        df[f'team_dnf_rate_season'] = df.groupby(['constructor', 'season'])['is_dnf'].transform(
            lambda x: x.expanding().mean().shift(1)
        )
        
        return df
    
    def _create_historical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on historical patterns
        """
        logger.info("Creating historical features...")
        
        # Championship position at time of race
        df = df.sort_values(['season', 'round', 'driver'])
        
        # Points accumulated so far in season
        df['season_points_so_far'] = df.groupby(['driver', 'season'])['points_earned'].transform('cumsum')
        df['season_position_avg'] = df.groupby(['driver', 'season'])['finishing_position'].transform(
            lambda x: x.expanding().mean()
        )
        
        # Momentum features
        df['recent_trend'] = df.groupby('driver')['finishing_position'].transform(
            lambda x: x.rolling(3).mean() - x.rolling(6).mean()
        )
        
        return df
    
    def _create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for different prediction tasks
        """
        logger.info("Creating target variables...")
        
        # Binary targets
        df['target_podium'] = (df['finishing_position'] <= 3).astype(int)
        df['target_points'] = (df['finishing_position'] <= 10).astype(int)
        df['target_win'] = (df['finishing_position'] == 1).astype(int)
        df['target_dnf'] = (~df['finishing_position'].notna()).astype(int)
        
        # Continuous targets
        df['target_position'] = df['finishing_position']
        df['target_points_earned'] = df['points_earned']
        
        # Position improvement
        df['target_position_gain'] = df['grid_position'] - df['finishing_position']
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Return list of engineered feature columns for model training
        """
        feature_columns = []
        
        # Form features
        feature_columns.extend([col for col in df.columns if col.startswith('form_')])
        
        # Circuit features
        feature_columns.extend([col for col in df.columns if col.startswith('circuit_')])
        
        # Driver features
        feature_columns.extend([col for col in df.columns if col.startswith('driver_')])
        
        # Team features
        feature_columns.extend([col for col in df.columns if col.startswith('team_')])
        
        # Historical features
        feature_columns.extend([col for col in df.columns if col.startswith('career_')])
        
        # Base features
        base_features = [
            'season', 'round', 'grid_position', 'grid_advantage', 
            'season_points_so_far', 'recent_trend'
        ]
        feature_columns.extend([col for col in base_features if col in df.columns])
        
        return feature_columns
    
    def get_target_columns(self) -> Dict[str, List[str]]:
        """
        Return mapping of prediction tasks to target columns
        """
        return {
            'classification': [
                'target_podium', 'target_points', 'target_win', 'target_dnf'
            ],
            'regression': [
                'target_position', 'target_points_earned', 'target_position_gain'
            ]
        }

# Example usage
if __name__ == "__main__":
    # Load consolidated data
    df = pd.read_csv("data/all_f1_results.csv")
    
    # Initialize feature engineering
    feature_engineer = F1FeatureEngineering()
    
    # Create features
    df_with_features = feature_engineer.create_all_features(df)
    
    # Save engineered dataset
    df_with_features.to_csv("data/f1_engineered_features.csv", index=False)
    
    # Show feature summary
    feature_cols = feature_engineer.get_feature_columns(df_with_features)
    target_cols = feature_engineer.get_target_columns()
    
    print(f"Created {len(feature_cols)} features")
    print(f"Classification targets: {target_cols['classification']}")
    print(f"Regression targets: {target_cols['regression']}")
