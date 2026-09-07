"""
Model Training Script
Trains the prediction models using historical MLB data
"""
import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.mlb_api_client import MLBAPIClient
from src.data_processor import DataProcessor
from src.ml_models import MLBPredictionModel
from config import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train prediction models with historical data"""
    
    def __init__(self):
        self.mlb_client = MLBAPIClient()
        self.data_processor = DataProcessor()
        self.ml_model = MLBPredictionModel()
        self.cfg = config['development']
    
    def generate_synthetic_training_data(self, num_samples: int = 500) -> tuple:
        """
        Generate synthetic training data for model development
        In production, this would use real historical game data from the API
        
        Args:
            num_samples: Number of training samples to generate
        
        Returns:
            Tuple of (X, y_winner, y_runs) DataFrames
        """
        logger.info(f"Generating {num_samples} synthetic training samples...")
        
        np.random.seed(42)
        
        data = []
        winners = []
        total_runs = []
        
        for i in range(num_samples):
            # Generate realistic feature ranges
            sample = {
                # Team Offensive Stats
                'home_runs_for': np.random.randint(3, 8),
                'home_runs_against': np.random.randint(2, 7),
                'home_avg_runs_per_game': np.random.uniform(3.5, 5.5),
                'away_runs_for': np.random.randint(3, 8),
                'away_runs_against': np.random.randint(2, 7),
                'away_avg_runs_per_game': np.random.uniform(3.5, 5.5),
                
                # Batting Stats
                'home_batting_avg': np.random.uniform(0.240, 0.280),
                'home_home_runs': np.random.randint(0, 3),
                'home_strikeouts': np.random.randint(5, 15),
                'home_walks': np.random.randint(2, 6),
                'home_ops': np.random.uniform(0.650, 0.850),
                'away_batting_avg': np.random.uniform(0.240, 0.280),
                'away_home_runs': np.random.randint(0, 3),
                'away_strikeouts': np.random.randint(5, 15),
                'away_walks': np.random.randint(2, 6),
                'away_ops': np.random.uniform(0.650, 0.850),
                
                # Pitching Stats
                'home_era': np.random.uniform(2.5, 5.0),
                'home_whip': np.random.uniform(1.0, 1.4),
                'home_k9': np.random.uniform(8, 12),
                'home_bb9': np.random.uniform(2, 4),
                'away_era': np.random.uniform(2.5, 5.0),
                'away_whip': np.random.uniform(1.0, 1.4),
                'away_k9': np.random.uniform(8, 12),
                'away_bb9': np.random.uniform(2, 4),
                
                # Record/Win %
                'home_wins': np.random.randint(40, 100),
                'home_losses': np.random.randint(40, 100),
                'home_win_pct': np.random.uniform(0.400, 0.650),
                'away_wins': np.random.randint(40, 100),
                'away_losses': np.random.randint(40, 100),
                'away_win_pct': np.random.uniform(0.400, 0.650),
                
                # Weather Features
                'temperature': np.random.uniform(50, 95),
                'feels_like': np.random.uniform(48, 100),
                'humidity': np.random.uniform(20, 95),
                'wind_speed': np.random.uniform(0, 15),
                'wind_direction': np.random.uniform(0, 360),
                'pressure': np.random.uniform(1000, 1020),
                'precipitation': np.random.uniform(0, 0.5),
                
                # Venue Features
                'altitude': np.random.choice([0, 500, 1000, 1500, 5000]),
                'capacity': np.random.randint(35000, 50000),
                'year_opened': np.random.randint(1950, 2020),
                'altitude_factor': 1.0,
                
                # Temporal Features
                'day_of_week': np.random.randint(0, 7),
                'month': np.random.randint(4, 11),
                'day_of_season': np.random.randint(1, 163),
                'season_phase': np.random.choice([0, 1, 2]),
                
                # Situational Features
                'rest_days_home': np.random.randint(0, 5),
                'rest_days_away': np.random.randint(0, 5),
                
                # Derived Features
                'run_differential_diff': np.random.uniform(-3, 3),
                'win_pct_diff': np.random.uniform(-0.2, 0.2),
                'era_diff': np.random.uniform(-2, 2),
                'batting_avg_diff': np.random.uniform(-0.030, 0.030),
                'home_field_advantage': 1.0,
                'expected_total_runs': np.random.uniform(6, 12),
            }
            
            # Calculate altitude factor
            if sample['altitude'] > 1500:
                sample['altitude_factor'] = 1.08
            elif sample['altitude'] > 1000:
                sample['altitude_factor'] = 1.04
            else:
                sample['altitude_factor'] = 1.0
            
            data.append(sample)
            
            # Generate target: Winner (based on run differential and win%)
            home_advantage = sample['run_differential_diff'] + (sample['win_pct_diff'] * 0.5)
            home_win_prob = 0.5 + (home_advantage * 0.1)
            winner = 1 if np.random.random() < home_win_prob else 0
            winners.append(winner)
            
            # Generate target: Total Runs
            base_runs = sample['home_avg_runs_per_game'] + sample['away_avg_runs_per_game']
            temp_factor = 1.0 + (sample['temperature'] - 70) * 0.01
            runs = base_runs * temp_factor * sample['altitude_factor'] + np.random.normal(0, 0.5)
            runs = max(1, min(20, runs))  # Clamp between 1 and 20
            total_runs.append(runs)
        
        df = pd.DataFrame(data)
        df_winners = pd.Series(winners, name='winner')
        df_runs = pd.Series(total_runs, name='total_runs')
        
        logger.info(f"Generated {len(df)} samples with {len(df.columns)} features")
        
        return df, df_winners, df_runs
    
    def train_models(self, X: pd.DataFrame, y_winner: pd.Series, y_runs: pd.Series):
        """Train both prediction models"""
        logger.info("Starting model training...")
        
        # Train winner prediction model
        logger.info("Training winner prediction model...")
        winner_metrics = self.ml_model.train_winner_model(X, y_winner, test_size=0.2)
        
        logger.info("Winner Model Metrics:")
        for metric, value in winner_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        # Train runs prediction model
        logger.info("Training runs prediction model...")
        runs_metrics = self.ml_model.train_runs_model(X, y_runs, test_size=0.2)
        
        logger.info("Runs Model Metrics:")
        for metric, value in runs_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return winner_metrics, runs_metrics
    
    def save_trained_models(self):
        """Save trained models to disk"""
        logger.info("Saving trained models...")
        
        os.makedirs('models', exist_ok=True)
        
        self.ml_model.save_models(
            self.cfg.MODEL_WINNER_PATH,
            self.cfg.MODEL_RUNS_PATH,
            self.cfg.SCALER_PATH
        )
        
        logger.info("Models saved successfully!")
    
    def display_feature_importance(self):
        """Display feature importance from trained models"""
        logger.info("\n" + "="*60)
        logger.info("WINNER MODEL - TOP 15 IMPORTANT FEATURES")
        logger.info("="*60)
        
        importance_winner = self.ml_model.get_feature_importance('winner')
        for idx, row in importance_winner.head(15).iterrows():
            logger.info(f"{row['feature']:30s} {row['importance']:8.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("RUNS MODEL - TOP 15 IMPORTANT FEATURES")
        logger.info("="*60)
        
        importance_runs = self.ml_model.get_feature_importance('runs')
        for idx, row in importance_runs.head(15).iterrows():
            logger.info(f"{row['feature']:30s} {row['importance']:8.4f}")
    
    def run(self):
        """Execute full training pipeline"""
        logger.info("="*60)
        logger.info("MLB PREDICTION AI - MODEL TRAINING")
        logger.info("="*60)
        
        # Generate training data
        X, y_winner, y_runs = self.generate_synthetic_training_data(num_samples=1000)
        
        logger.info(f"\nDataset Summary:")
        logger.info(f"  Training samples: {len(X)}")
        logger.info(f"  Features: {len(X.columns)}")
        logger.info(f"  Winner distribution: {y_winner.value_counts().to_dict()}")
        logger.info(f"  Total runs range: {y_runs.min():.1f} - {y_runs.max():.1f}")
        
        # Train models
        winner_metrics, runs_metrics = self.train_models(X, y_winner, y_runs)
        
        # Save models
        self.save_trained_models()
        
        # Display feature importance
        self.display_feature_importance()
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Winner Model Accuracy: {winner_metrics['accuracy']:.4f}")
        logger.info(f"Runs Model RMSE: {runs_metrics['rmse']:.4f}")
        logger.info(f"Models saved to: ./models/")
        logger.info("\nNext steps:")
        logger.info("1. Configure .env file with your API keys")
        logger.info("2. Run: python app.py")
        logger.info("3. Test predictions at: http://localhost:5000/predict")
        logger.info("="*60)

if __name__ == '__main__':
    trainer = ModelTrainer()
    trainer.run()
