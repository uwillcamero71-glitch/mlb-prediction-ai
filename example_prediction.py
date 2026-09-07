"""
Example Script - MLB Prediction Demo
Demonstrates how to use the prediction models
"""
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.data_processor import DataProcessor
from src.ml_models import MLBPredictionModel
import pandas as pd

def main():
    print("="*70)
    print("🎲 MLB PREDICTION AI - DEMO")
    print("="*70)
    
    # Initialize components
    data_processor = DataProcessor()
    ml_model = MLBPredictionModel()
    
    # Try to load pre-trained models
    try:
        ml_model.load_models(
            'models/winner_model.pkl',
            'models/runs_model.pkl',
            'models/scaler.pkl'
        )
        print("\n✅ Models loaded successfully!")
    except FileNotFoundError:
        print("\n⚠️  Models not found. Please run 'python train_models.py' first")
        print("\nRunning training now...\n")
        
        # Train models if they don't exist
        from train_models import ModelTrainer
        trainer = ModelTrainer()
        trainer.run()
    
    # Example 1: New York Yankees (Home) vs Boston Red Sox (Away)
    print("\n" + "="*70)
    print("EXAMPLE 1: New York Yankees (Home) vs Boston Red Sox (Away)")
    print("="*70)
    
    home_stats = {
        'runs_scored': 715,
        'runs_allowed': 620,
        'batting_avg': 0.265,
        'home_runs': 180,
        'strikeouts': 1200,
        'walks': 550,
        'ops': 0.758,
        'era': 3.85,
        'whip': 1.15,
        'k9': 9.2,
        'bb9': 2.8,
        'wins': 85,
        'losses': 65,
        'avg_runs_per_game': 4.4,
        'rest_days': 1
    }
    
    away_stats = {
        'runs_scored': 680,
        'runs_allowed': 640,
        'batting_avg': 0.255,
        'home_runs': 165,
        'strikeouts': 1300,
        'walks': 480,
        'ops': 0.720,
        'era': 4.05,
        'whip': 1.22,
        'k9': 8.8,
        'bb9': 3.0,
        'wins': 80,
        'losses': 70,
        'avg_runs_per_game': 4.2,
        'rest_days': 1
    }
    
    weather = {
        'temperature': 75,
        'feels_like': 74,
        'humidity': 65,
        'wind_speed': 8,
        'wind_direction': 180,
        'pressure': 1013,
        'rain': 0
    }
    
    venue = {
        'altitude': 20,
        'capacity': 47309,
        'year_opened': 2009
    }
    
    # Process game data
    game_features = data_processor.process_game_data(
        home_team='New York Yankees',
        away_team='Boston Red Sox',
        game_date='2024-09-10T19:05:00Z',
        home_stats=home_stats,
        away_stats=away_stats,
        weather=weather,
        venue_info=venue
    )
    
    print("\n📊 Game Features Generated:")
    print(f"  Home Team Advantage: YES (Yankee Stadium)")
    print(f"  Temperature: {weather['temperature']}°F")
    print(f"  Wind Speed: {weather['wind_speed']} mph")
    print(f"  Altitude: {venue['altitude']} ft (Sea level)")
    
    # Make predictions
    try:
        prediction = ml_model.predict_game(game_features)
        
        print("\n🎯 PREDICTIONS:")
        print(f"\n  Winner Prediction:")
        print(f"    - Predicted Winner: {prediction['winner_prediction']['prediction'].upper()}")
        print(f"    - Confidence: {prediction['winner_prediction']['confidence']:.1f}%")
        print(f"    - Home Win Probability: {prediction['winner_prediction']['probabilities']['home_win']:.1f}%")
        print(f"    - Away Win Probability: {prediction['winner_prediction']['probabilities']['away_win']:.1f}%")
        
        print(f"\n  Runs Prediction:")
        print(f"    - Expected Total Runs: {prediction['runs_prediction']['predicted_runs']}")
        print(f"    - Range: {prediction['runs_prediction']['lower_bound']} - {prediction['runs_prediction']['upper_bound']}")
        print(f"    - Over/Under 8: {prediction['runs_prediction']['over_under_8']}")
        
        print(f"\n  Combined Analysis:")
        print(f"    - Likely Outcome: {prediction['combined_analysis']['likely_outcome']}")
        print(f"    - Confidence Level: {prediction['combined_analysis']['confidence_level']}")
        
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
        print("Make sure models are trained first!")
    
    # Example 2: Colorado Rockies (Home) vs Los Angeles Dodgers (Away)
    print("\n" + "="*70)
    print("EXAMPLE 2: Colorado Rockies (Home) vs Los Angeles Dodgers (Away)")
    print("="*70)
    
    home_stats_2 = {
        'runs_scored': 720,
        'runs_allowed': 680,
        'batting_avg': 0.268,
        'home_runs': 195,
        'strikeouts': 1150,
        'walks': 560,
        'ops': 0.775,
        'era': 4.15,
        'whip': 1.25,
        'k9': 8.9,
        'bb9': 3.1,
        'wins': 82,
        'losses': 68,
        'avg_runs_per_game': 4.5,
        'rest_days': 2
    }
    
    away_stats_2 = {
        'runs_scored': 735,
        'runs_allowed': 610,
        'batting_avg': 0.275,
        'home_runs': 210,
        'strikeouts': 1100,
        'walks': 600,
        'ops': 0.815,
        'era': 3.65,
        'whip': 1.10,
        'k9': 9.5,
        'bb9': 2.6,
        'wins': 92,
        'losses': 58,
        'avg_runs_per_game': 4.6,
        'rest_days': 1
    }
    
    weather_2 = {
        'temperature': 82,
        'feels_like': 85,
        'humidity': 40,
        'wind_speed': 12,
        'wind_direction': 270,
        'pressure': 1005,
        'rain': 0
    }
    
    venue_2 = {
        'altitude': 5277,
        'capacity': 50430,
        'year_opened': 1995
    }
    
    game_features_2 = data_processor.process_game_data(
        home_team='Colorado Rockies',
        away_team='Los Angeles Dodgers',
        game_date='2024-09-11T20:10:00Z',
        home_stats=home_stats_2,
        away_stats=away_stats_2,
        weather=weather_2,
        venue_info=venue_2
    )
    
    print("\n📊 Game Features Generated:")
    print(f"  Home Team: Colorado Rockies (High Altitude Advantage!)")
    print(f"  Temperature: {weather_2['temperature']}°F")
    print(f"  Wind Speed: {weather_2['wind_speed']} mph")
    print(f"  Altitude: {venue_2['altitude']} ft ⭐ (Coors Field - +8% Home Runs!)")
    
    try:
        prediction_2 = ml_model.predict_game(game_features_2)
        
        print("\n🎯 PREDICTIONS:")
        print(f"\n  Winner Prediction:")
        print(f"    - Predicted Winner: {prediction_2['winner_prediction']['prediction'].upper()}")
        print(f"    - Confidence: {prediction_2['winner_prediction']['confidence']:.1f}%")
        print(f"    - Home Win Probability: {prediction_2['winner_prediction']['probabilities']['home_win']:.1f}%")
        print(f"    - Away Win Probability: {prediction_2['winner_prediction']['probabilities']['away_win']:.1f}%")
        
        print(f"\n  Runs Prediction:")
        print(f"    - Expected Total Runs: {prediction_2['runs_prediction']['predicted_runs']}")
        print(f"    - Range: {prediction_2['runs_prediction']['lower_bound']} - {prediction_2['runs_prediction']['upper_bound']}")
        print(f"    - Over/Under 8: {prediction_2['runs_prediction']['over_under_8']}")
        print(f"    - Note: High altitude = more runs expected!")
        
        print(f"\n  Combined Analysis:")
        print(f"    - Likely Outcome: {prediction_2['combined_analysis']['likely_outcome']}")
        print(f"    - Confidence Level: {prediction_2['combined_analysis']['confidence_level']}")
        
    except Exception as e:
        print(f"\n❌ Prediction error: {e}")
    
    print("\n" + "="*70)
    print("✨ DEMO COMPLETE")
    print("="*70)
    print("\nNext steps:")
    print("1. Configure .env file with your OpenWeatherMap API key")
    print("2. Run: python app.py")
    print("3. Test the REST API at: http://localhost:5000")
    print("4. Use POST /predict endpoint for game predictions")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()
