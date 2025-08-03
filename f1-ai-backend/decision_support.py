"""
F1 Decision Support System
Provides actionable insights and strategic recommendations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

class F1DecisionSupport:
    def __init__(self):
        self.circuit_strategies = {
            'monaco': {
                'qualifying_importance': 10,  # Scale 1-10
                'pit_window_optimal': [35, 45],
                'weather_sensitivity': 8,
                'tire_degradation_factor': 0.3,
                'safety_car_probability': 0.6
            },
            'silverstone': {
                'qualifying_importance': 7,
                'pit_window_optimal': [25, 35],
                'weather_sensitivity': 9,
                'tire_degradation_factor': 0.7,
                'safety_car_probability': 0.2
            },
            'spa': {
                'qualifying_importance': 6,
                'pit_window_optimal': [20, 30],
                'weather_sensitivity': 9,
                'tire_degradation_factor': 0.6,
                'safety_car_probability': 0.3
            },
            'monza': {
                'qualifying_importance': 5,
                'pit_window_optimal': [22, 32],
                'weather_sensitivity': 4,
                'tire_degradation_factor': 0.5,
                'safety_car_probability': 0.2
            }
        }
    
    def analyze_driver_circuit_performance(self, driver: str, circuit: str, 
                                         historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze driver's historical performance at specific circuit
        """
        logger.info(f"Analyzing {driver} performance at {circuit}")
        
        # Filter data for driver and circuit
        driver_circuit_data = historical_data[
            (historical_data['driver'] == driver) & 
            (historical_data['circuit_name'] == circuit.lower())
        ].copy()
        
        if driver_circuit_data.empty:
            return {
                'circuit': circuit,
                'driver': driver,
                'historical_races': 0,
                'avg_position': None,
                'best_position': None,
                'podium_rate': 0,
                'dnf_rate': 0,
                'performance_trend': 'insufficient_data',
                'recommendations': ['Insufficient historical data for analysis']
            }
        
        # Calculate statistics
        total_races = len(driver_circuit_data)
        valid_finishes = driver_circuit_data['finishing_position'].dropna()
        
        if len(valid_finishes) > 0:
            avg_position = valid_finishes.mean()
            best_position = valid_finishes.min()
            podium_count = (valid_finishes <= 3).sum()
            podium_rate = podium_count / len(valid_finishes)
        else:
            avg_position = None
            best_position = None
            podium_rate = 0
        
        dnf_count = driver_circuit_data['finishing_position'].isna().sum()
        dnf_rate = dnf_count / total_races if total_races > 0 else 0
        
        # Analyze trend (recent vs historical performance)
        if len(driver_circuit_data) >= 3:
            recent_races = driver_circuit_data.tail(3)['finishing_position'].dropna()
            historical_races = driver_circuit_data.head(-3)['finishing_position'].dropna()
            
            if len(recent_races) > 0 and len(historical_races) > 0:
                recent_avg = recent_races.mean()
                historical_avg = historical_races.mean()
                
                if recent_avg < historical_avg - 1:
                    trend = 'improving'
                elif recent_avg > historical_avg + 1:
                    trend = 'declining'
                else:
                    trend = 'stable'
            else:
                trend = 'insufficient_data'
        else:
            trend = 'insufficient_data'
        
        # Generate recommendations
        recommendations = self._generate_circuit_recommendations(
            driver, circuit, avg_position, podium_rate, dnf_rate, trend
        )
        
        return {
            'circuit': circuit,
            'driver': driver,
            'historical_races': total_races,
            'avg_position': round(avg_position, 2) if avg_position else None,
            'best_position': int(best_position) if best_position else None,
            'podium_rate': round(podium_rate, 3),
            'dnf_rate': round(dnf_rate, 3),
            'performance_trend': trend,
            'recommendations': recommendations
        }
    
    def _generate_circuit_recommendations(self, driver: str, circuit: str, 
                                        avg_position: float, podium_rate: float, 
                                        dnf_rate: float, trend: str) -> List[str]:
        """
        Generate strategic recommendations based on performance analysis
        """
        recommendations = []
        
        circuit_info = self.circuit_strategies.get(circuit.lower(), {})
        
        # Qualifying strategy
        qualifying_importance = circuit_info.get('qualifying_importance', 5)
        if qualifying_importance >= 8:
            recommendations.append(f"HIGH PRIORITY: Qualifying position crucial at {circuit}. Focus maximum effort on Q3.")
        elif avg_position and avg_position > 10:
            recommendations.append(f"Qualifying focus needed - average position of {avg_position:.1f} suggests grid position issues")
        
        # Performance trend analysis
        if trend == 'improving':
            recommendations.append(f"Positive trend at {circuit} - continue current setup approach")
        elif trend == 'declining':
            recommendations.append(f"Performance declining at {circuit} - consider setup changes or strategy revision")
        
        # DNF analysis
        if dnf_rate > 0.2:
            recommendations.append(f"High DNF rate ({dnf_rate:.1%}) at {circuit} - review reliability and risk management")
        
        # Circuit-specific strategies
        if circuit.lower() == 'monaco':
            recommendations.append("Monaco strategy: Prioritize track position over pace. Consider aggressive undercut opportunities.")
        elif circuit.lower() == 'silverstone':
            recommendations.append("Silverstone strategy: Weather contingency crucial. Prepare for multiple tire compound strategies.")
        elif circuit.lower() == 'spa':
            recommendations.append("Spa strategy: DRS efficiency key. Balance downforce for straights vs. sector 2 performance.")
        
        return recommendations
    
    def optimize_qualifying_strategy(self, driver: str, circuit: str, 
                                   weather_forecast: Dict[str, Any],
                                   grid_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Recommend optimal qualifying strategy based on predictions and conditions
        """
        logger.info(f"Optimizing qualifying strategy for {driver} at {circuit}")
        
        circuit_info = self.circuit_strategies.get(circuit.lower(), {})
        qualifying_importance = circuit_info.get('qualifying_importance', 5)
        weather_sensitivity = circuit_info.get('weather_sensitivity', 5)
        
        # Find driver's predicted grid position
        driver_prediction = next((p for p in grid_predictions if p['driver'] == driver), None)
        predicted_grid = driver_prediction['grid_position'] if driver_prediction else 10
        
        strategy = {
            'driver': driver,
            'circuit': circuit,
            'predicted_grid': predicted_grid,
            'qualifying_importance_score': qualifying_importance,
            'weather_risk_score': weather_sensitivity,
            'recommendations': []
        }
        
        # Weather-based recommendations
        rain_probability = weather_forecast.get('rain_probability', 0)
        if rain_probability > 0.3 and weather_sensitivity >= 7:
            strategy['recommendations'].append("HIGH RISK: Rain likely and circuit weather-sensitive. Prioritize early Q3 runs.")
            strategy['tire_strategy'] = 'aggressive_early'
        elif rain_probability > 0.5:
            strategy['recommendations'].append("Weather wildcard opportunity: Rain could shuffle grid. Prepare for mixed conditions.")
            strategy['tire_strategy'] = 'flexible'
        else:
            strategy['tire_strategy'] = 'conventional'
        
        # Grid position strategy
        if predicted_grid <= 3:
            strategy['recommendations'].append("Top 3 grid predicted: Focus on pole position for race advantage.")
            strategy['session_approach'] = 'maximize_performance'
        elif predicted_grid <= 6:
            strategy['recommendations'].append("Q3 achievable: Balance tire saving with performance for optimal grid position.")
            strategy['session_approach'] = 'balanced'
        elif predicted_grid <= 10:
            strategy['recommendations'].append("Q3 marginal: Consider aggressive approach to break into top 10.")
            strategy['session_approach'] = 'aggressive'
        else:
            strategy['recommendations'].append("Focus on Q2 progression and race setup optimization.")
            strategy['session_approach'] = 'race_focused'
        
        # Circuit-specific qualifying advice
        if qualifying_importance >= 8:
            strategy['recommendations'].append(f"CRITICAL: Grid position determines race outcome at {circuit}. Maximize qualifying performance.")
        
        return strategy
    
    def predict_pit_window_optimization(self, race_predictions: List[Dict[str, Any]], 
                                      circuit: str, weather_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict optimal pit stop windows and strategies
        """
        logger.info(f"Analyzing pit window optimization for {circuit}")
        
        circuit_info = self.circuit_strategies.get(circuit.lower(), {})
        base_pit_window = circuit_info.get('pit_window_optimal', [25, 35])
        tire_degradation = circuit_info.get('tire_degradation_factor', 0.5)
        safety_car_prob = circuit_info.get('safety_car_probability', 0.3)
        
        # Weather adjustments
        temperature = weather_conditions.get('track_temperature', 35)
        rain_risk = weather_conditions.get('rain_probability', 0)
        
        # Adjust pit window for conditions
        if temperature > 40:  # Hot conditions increase degradation
            adjusted_window = [w - 3 for w in base_pit_window]
            strategy_note = "Hot track conditions: Earlier pit window recommended due to increased tire degradation"
        elif rain_risk > 0.3:
            adjusted_window = [w - 5 for w in base_pit_window]
            strategy_note = "Rain risk: Consider earlier pit window for weather contingency"
        else:
            adjusted_window = base_pit_window
            strategy_note = "Standard pit window conditions"
        
        # Safety car strategy
        if safety_car_prob > 0.5:
            sc_strategy = "High safety car probability: Prepare for opportunistic pit stops during SC periods"
        else:
            sc_strategy = "Low safety car risk: Plan conventional pit strategy"
        
        # Driver-specific recommendations
        driver_strategies = []
        for prediction in race_predictions[:6]:  # Top 6 drivers
            driver = prediction['driver']
            predicted_pos = prediction.get('predicted_position', 10)
            
            if predicted_pos <= 3:
                driver_strategies.append({
                    'driver': driver,
                    'strategy': 'defensive',
                    'pit_window': adjusted_window,
                    'notes': 'Protect track position, react to competitors'
                })
            elif predicted_pos <= 8:
                driver_strategies.append({
                    'driver': driver,
                    'strategy': 'aggressive',
                    'pit_window': [w - 2 for w in adjusted_window],
                    'notes': 'Undercut opportunity to gain positions'
                })
            else:
                driver_strategies.append({
                    'driver': driver,
                    'strategy': 'alternative',
                    'pit_window': [w + 5 for w in adjusted_window],
                    'notes': 'Extended stint for strategic advantage'
                })
        
        return {
            'circuit': circuit,
            'base_pit_window': base_pit_window,
            'adjusted_pit_window': adjusted_window,
            'strategy_note': strategy_note,
            'safety_car_strategy': sc_strategy,
            'tire_degradation_factor': tire_degradation,
            'driver_strategies': driver_strategies,
            'weather_impact': {
                'temperature': temperature,
                'rain_risk': rain_risk,
                'degradation_multiplier': 1 + (temperature - 35) * 0.02
            }
        }
    
    def identify_underperforming_circuits(self, driver: str, 
                                        historical_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify circuits where driver underperforms relative to their average
        """
        logger.info(f"Identifying underperforming circuits for {driver}")
        
        driver_data = historical_data[historical_data['driver'] == driver].copy()
        
        if driver_data.empty:
            return []
        
        # Calculate overall driver average
        overall_avg = driver_data['finishing_position'].dropna().mean()
        
        # Group by circuit and calculate averages
        circuit_performance = driver_data.groupby('circuit_name').agg({
            'finishing_position': ['mean', 'count', 'std'],
            'grid_position': 'mean'
        }).round(2)
        
        circuit_performance.columns = ['avg_finish', 'race_count', 'finish_std', 'avg_grid']
        circuit_performance = circuit_performance.reset_index()
        
        # Identify underperforming circuits
        underperforming = []
        for _, row in circuit_performance.iterrows():
            if row['race_count'] >= 2:  # Need at least 2 races for valid comparison
                performance_gap = row['avg_finish'] - overall_avg
                
                if performance_gap > 2:  # Significantly worse than average
                    underperforming.append({
                        'circuit': row['circuit_name'],
                        'avg_finish_position': row['avg_finish'],
                        'driver_overall_avg': round(overall_avg, 2),
                        'performance_gap': round(performance_gap, 2),
                        'races_analyzed': int(row['race_count']),
                        'consistency': round(row['finish_std'], 2),
                        'avg_grid_position': row['avg_grid'],
                        'severity': 'high' if performance_gap > 4 else 'moderate'
                    })
        
        # Sort by performance gap (worst first)
        underperforming.sort(key=lambda x: x['performance_gap'], reverse=True)
        
        return underperforming
    
    def generate_race_strategy_report(self, driver: str, circuit: str,
                                    race_predictions: List[Dict[str, Any]],
                                    weather_forecast: Dict[str, Any],
                                    historical_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive race strategy report
        """
        logger.info(f"Generating race strategy report for {driver} at {circuit}")
        
        # Get all analyses
        circuit_analysis = self.analyze_driver_circuit_performance(driver, circuit, historical_data)
        qualifying_strategy = self.optimize_qualifying_strategy(driver, circuit, weather_forecast, race_predictions)
        pit_strategy = self.predict_pit_window_optimization(race_predictions, circuit, weather_forecast)
        underperforming_circuits = self.identify_underperforming_circuits(driver, historical_data)
        
        # Find driver's race prediction
        driver_prediction = next((p for p in race_predictions if p['driver'] == driver), None)
        
        # Compile comprehensive report
        report = {
            'report_generated': datetime.now().isoformat(),
            'driver': driver,
            'circuit': circuit,
            'race_prediction': driver_prediction,
            'historical_performance': circuit_analysis,
            'qualifying_strategy': qualifying_strategy,
            'pit_stop_strategy': pit_strategy,
            'key_insights': [],
            'strategic_priorities': [],
            'risk_factors': [],
            'opportunities': []
        }
        
        # Generate key insights
        if circuit_analysis['avg_position'] and circuit_analysis['avg_position'] < 5:
            report['key_insights'].append(f"Strong historical performance at {circuit} (avg: P{circuit_analysis['avg_position']:.1f})")
        
        if circuit_analysis['performance_trend'] == 'improving':
            report['opportunities'].append("Positive performance trend at this circuit")
        elif circuit_analysis['performance_trend'] == 'declining':
            report['risk_factors'].append("Recent performance decline at this circuit")
        
        # Strategic priorities
        if qualifying_strategy['qualifying_importance_score'] >= 8:
            report['strategic_priorities'].append("CRITICAL: Maximize qualifying performance")
        
        if weather_forecast.get('rain_probability', 0) > 0.3:
            report['strategic_priorities'].append("Weather contingency planning essential")
            report['risk_factors'].append(f"Rain probability: {weather_forecast.get('rain_probability', 0):.0%}")
        
        # Check if this is an underperforming circuit
        circuit_issues = next((c for c in underperforming_circuits if c['circuit'] == circuit.lower()), None)
        if circuit_issues:
            report['risk_factors'].append(f"Historically underperforming circuit (P{circuit_issues['avg_finish_position']:.1f} vs P{circuit_issues['driver_overall_avg']:.1f} average)")
            report['strategic_priorities'].append("Address circuit-specific weaknesses")
        
        return report
    
    def export_strategy_insights(self, reports: List[Dict[str, Any]], 
                               filename: str = None) -> str:
        """
        Export strategy insights to JSON file
        """
        if filename is None:
            filename = f"data/f1_strategy_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w') as f:
            json.dump(reports, f, indent=2, default=str)
        
        logger.info(f"Strategy insights exported to {filename}")
        return filename

# Example usage
if __name__ == "__main__":
    # Initialize decision support system
    decision_support = F1DecisionSupport()
    
    # Load historical data (placeholder)
    # historical_data = pd.read_csv("data/f1_engineered_features.csv")
    
    # Example race predictions
    race_predictions = [
        {'driver': 'Max Verstappen', 'predicted_position': 1, 'grid_position': 1},
        {'driver': 'Lando Norris', 'predicted_position': 2, 'grid_position': 3},
        {'driver': 'Charles Leclerc', 'predicted_position': 3, 'grid_position': 2}
    ]
    
    # Example weather forecast
    weather_forecast = {
        'track_temperature': 35,
        'air_temperature': 28,
        'rain_probability': 0.2,
        'humidity': 60
    }
    
    # Generate pit strategy analysis
    pit_analysis = decision_support.predict_pit_window_optimization(
        race_predictions, 'silverstone', weather_forecast
    )
    
    print("Pit Strategy Analysis:")
    print(json.dumps(pit_analysis, indent=2, default=str))
