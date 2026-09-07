"""
Data Processing Module
Handles feature engineering and data preparation for ML models
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and engineer features for ML models"""
    
    def __init__(self):
        self.feature_columns = None
        self.categorical_features = ['day_of_week', 'month', 'season_phase']
    
    def process_game_data(self, 
                         home_team: str, 
                         away_team: str,
                         game_date: str,
                         home_stats: Dict,
                         away_stats: Dict,
                         weather: Dict,
                         venue_info: Dict) -> pd.DataFrame:
        """
        Process all game data into features for prediction
        
        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game datetime
            home_stats: Home team statistics
            away_stats: Away team statistics
            weather: Weather data
            venue_info: Stadium/venue information
        
        Returns:
            DataFrame with engineered features
        """
        features = {}
        
        # Parse date
        game_datetime = pd.to_datetime(game_date)
        
        # === Team Offensive Stats ===
        features['home_runs_for'] = home_stats.get('runs_scored', 0)
        features['home_runs_against'] = home_stats.get('runs_allowed', 0)
        features['home_run_differential'] = features['home_runs_for'] - features['home_runs_against']
        features['home_avg_runs_per_game'] = home_stats.get('avg_runs_per_game', 0)
        
        features['away_runs_for'] = away_stats.get('runs_scored', 0)
        features['away_runs_against'] = away_stats.get('runs_allowed', 0)
        features['away_run_differential'] = features['away_runs_for'] - features['away_runs_against']
        features['away_avg_runs_per_game'] = away_stats.get('avg_runs_per_game', 0)
        
        # === Batting Stats ===
        features['home_batting_avg'] = home_stats.get('batting_avg', 0)
        features['home_home_runs'] = home_stats.get('home_runs', 0)
        features['home_strikeouts'] = home_stats.get('strikeouts', 0)
        features['home_walks'] = home_stats.get('walks', 0)
        features['home_ops'] = home_stats.get('ops', 0)
        
        features['away_batting_avg'] = away_stats.get('batting_avg', 0)
        features['away_home_runs'] = away_stats.get('home_runs', 0)
        features['away_strikeouts'] = away_stats.get('strikeouts', 0)
        features['away_walks'] = away_stats.get('walks', 0)
        features['away_ops'] = away_stats.get('ops', 0)
        
        # === Pitching Stats ===
        features['home_era'] = home_stats.get('era', 0)
        features['home_whip'] = home_stats.get('whip', 0)
        features['home_k9'] = home_stats.get('k9', 0)
        features['home_bb9'] = home_stats.get('bb9', 0)
        
        features['away_era'] = away_stats.get('era', 0)
        features['away_whip'] = away_stats.get('whip', 0)
        features['away_k9'] = away_stats.get('k9', 0)
        features['away_bb9'] = away_stats.get('bb9', 0)
        
        # === Record/Win % ===
        features['home_wins'] = home_stats.get('wins', 0)
        features['home_losses'] = home_stats.get('losses', 0)
        features['home_win_pct'] = features['home_wins'] / (features['home_wins'] + features['home_losses']) if (features['home_wins'] + features['home_losses']) > 0 else 0
        
        features['away_wins'] = away_stats.get('wins', 0)
        features['away_losses'] = away_stats.get('losses', 0)
        features['away_win_pct'] = features['away_wins'] / (features['away_wins'] + features['away_losses']) if (features['away_wins'] + features['away_losses']) > 0 else 0
        
        # === Weather Features ===
        features['temperature'] = weather.get('temperature', 0)
        features['feels_like'] = weather.get('feels_like', 0)
        features['humidity'] = weather.get('humidity', 0)
        features['wind_speed'] = weather.get('wind_speed', 0)
        features['wind_direction'] = weather.get('wind_direction', 0)
        features['pressure'] = weather.get('pressure', 1013)
        features['precipitation'] = weather.get('rain', 0)
        
        # === Venue Features ===
        features['altitude'] = venue_info.get('altitude', 0)
        features['capacity'] = venue_info.get('capacity', 0)
        features['year_opened'] = venue_info.get('year_opened', 2000)
        
        # Calculate altitude factor (affects runs)
        features['altitude_factor'] = 1.0 if features['altitude'] < 1000 else 1.08 if features['altitude'] > 1500 else 1.04
        
        # === Temporal Features ===
        features['day_of_week'] = game_datetime.dayofweek
        features['month'] = game_datetime.month
        features['day_of_season'] = game_datetime.dayofyear
        
        # Determine season phase
        if game_datetime.month in [4, 5]:
            features['season_phase'] = 'early'  # 0
        elif game_datetime.month in [6, 7, 8]:
            features['season_phase'] = 'mid'  # 1
        else:
            features['season_phase'] = 'late'  # 2
        
        # === Situational Features ===
        features['rest_days_home'] = home_stats.get('rest_days', 1)
        features['rest_days_away'] = away_stats.get('rest_days', 1)
        
        # === Derived Features ===
        features['run_differential_diff'] = features['home_run_differential'] - features['away_run_differential']
        features['win_pct_diff'] = features['home_win_pct'] - features['away_win_pct']
        features['era_diff'] = features['away_era'] - features['home_era']
        features['batting_avg_diff'] = features['home_batting_avg'] - features['away_batting_avg']
        
        # Home field advantage factor
        features['home_field_advantage'] = 1.0
        
        # Total runs estimate based on weather
        features['expected_total_runs'] = self._calculate_expected_runs(
            features['home_avg_runs_per_game'],
            features['away_avg_runs_per_game'],
            features['temperature'],
            features['humidity'],
            features['wind_speed'],
            features['altitude_factor']
        )
        
        return pd.DataFrame([features])
    
    def _calculate_expected_runs(self, 
                                 home_avg: float,
                                 away_avg: float,
                                 temp: float,
                                 humidity: float,
                                 wind_speed: float,
                                 altitude_factor: float) -> float:
        """
        Calculate expected total runs based on weather and team averages
        
        Temperature impact: Higher temp = more home runs
        Humidity: Lower humidity = faster ball movement
        Wind: Outfield wind direction matters
        Altitude: Higher altitude = ball travels further
        """
        base_runs = home_avg + away_avg
        
        # Temperature adjustment (reference 70°F / 21°C)
        temp_adjustment = 1.0 + (temp - 21) * 0.01
        temp_adjustment = max(0.95, min(1.08, temp_adjustment))  # Clamp between 0.95 and 1.08
        
        # Humidity adjustment (lower humidity = faster ball)
        humidity_adjustment = 1.0 - (humidity / 100) * 0.02
        humidity_adjustment = max(0.98, min(1.02, humidity_adjustment))
        
        # Wind adjustment (simplified - strong winds affect home runs)
        wind_adjustment = 1.0 + (wind_speed * 0.005)
        wind_adjustment = max(0.98, min(1.05, wind_adjustment))
        
        expected_runs = base_runs * temp_adjustment * humidity_adjustment * wind_adjustment * altitude_factor
        
        return round(expected_runs, 2)
    
    def normalize_features(self, df: pd.DataFrame, scaler=None):
        """
        Normalize features using StandardScaler
        
        Args:
            df: DataFrame with features
            scaler: Fitted scaler (if None, creates new one)
        
        Returns:
            Normalized DataFrame and scaler object
        """
        from sklearn.preprocessing import StandardScaler
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
        
        if scaler is None:
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(df[numeric_cols])
        else:
            normalized_data = scaler.transform(df[numeric_cols])
        
        df_normalized = pd.DataFrame(normalized_data, columns=numeric_cols)
        
        return df_normalized, scaler
    
    def encode_categorical_features(self, df: pd.DataFrame):
        """
        Encode categorical features
        
        Args:
            df: DataFrame with categorical features
        
        Returns:
            DataFrame with encoded features
        """
        from sklearn.preprocessing import LabelEncoder
        
        df_encoded = df.copy()
        
        for col in self.categorical_features:
            if col in df_encoded.columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col])
        
        return df_encoded
    
    def create_feature_matrix(self, games: List[Dict]) -> pd.DataFrame:
        """
        Create feature matrix from multiple games
        
        Args:
            games: List of game dictionaries with all data
        
        Returns:
            DataFrame with all features for all games
        """
        features_list = []
        
        for game in games:
            features = self.process_game_data(
                game['home_team'],
                game['away_team'],
                game['date'],
                game['home_stats'],
                game['away_stats'],
                game['weather'],
                game['venue']
            )
            features_list.append(features)
        
        return pd.concat(features_list, ignore_index=True)
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        return [
            'home_runs_for', 'home_runs_against', 'home_run_differential',
            'home_avg_runs_per_game', 'away_runs_for', 'away_runs_against',
            'away_run_differential', 'away_avg_runs_per_game',
            'home_batting_avg', 'home_home_runs', 'home_strikeouts', 'home_walks', 'home_ops',
            'away_batting_avg', 'away_home_runs', 'away_strikeouts', 'away_walks', 'away_ops',
            'home_era', 'home_whip', 'home_k9', 'home_bb9',
            'away_era', 'away_whip', 'away_k9', 'away_bb9',
            'home_wins', 'home_losses', 'home_win_pct',
            'away_wins', 'away_losses', 'away_win_pct',
            'temperature', 'feels_like', 'humidity', 'wind_speed', 'wind_direction',
            'pressure', 'precipitation', 'altitude', 'capacity', 'year_opened',
            'altitude_factor', 'day_of_week', 'month', 'day_of_season',
            'rest_days_home', 'rest_days_away', 'run_differential_diff',
            'win_pct_diff', 'era_diff', 'batting_avg_diff', 'home_field_advantage',
            'expected_total_runs'
        ]
