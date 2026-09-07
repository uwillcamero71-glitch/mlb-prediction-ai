"""
Flask API for MLB Predictions
Main application with prediction endpoints
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import logging
from datetime import datetime
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.mlb_api_client import MLBAPIClient, WeatherClient
from src.data_processor import DataProcessor
from src.ml_models import MLBPredictionModel
from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config.from_object(config[os.getenv('FLASK_ENV', 'development')])

# Initialize components
mlb_client = MLBAPIClient()
weather_client = WeatherClient(app.config['WEATHER_API_KEY'])
data_processor = DataProcessor()
ml_model = MLBPredictionModel()

# Global cache for team stats
team_stats_cache = {}

# ============== UTILITY FUNCTIONS ==============

def get_team_info(team_id: int):
    """Get cached team info or fetch from API"""
    if team_id not in team_stats_cache:
        team_stats_cache[team_id] = mlb_client.get_season_stats(team_id)
    return team_stats_cache[team_id]

def extract_stats_from_api(stats_dict: dict) -> dict:
    """Extract relevant statistics from API response"""
    hitting = stats_dict.get('hitting', {})
    pitching = stats_dict.get('pitching', {})
    
    return {
        'runs_scored': hitting.get('runs', 0),
        'runs_allowed': pitching.get('runs', 0),
        'batting_avg': hitting.get('avg', 0),
        'home_runs': hitting.get('homeRuns', 0),
        'strikeouts': hitting.get('strikeOuts', 0),
        'walks': hitting.get('walks', 0),
        'ops': hitting.get('ops', 0),
        'era': pitching.get('era', 0),
        'whip': pitching.get('whip', 0),
        'k9': pitching.get('strikeOuts', 0),
        'bb9': pitching.get('walks', 0),
        'wins': 0,
        'losses': 0,
        'avg_runs_per_game': hitting.get('runs', 0) / 162 if hitting.get('runs') else 0,
        'rest_days': 1
    }

def get_stadium_info(venue_id: int) -> dict:
    """Get stadium information"""
    stadiums = {
        1: {'name': 'Yankee Stadium', 'altitude': 20, 'capacity': 47309, 'year_opened': 2009},
        2: {'name': 'Rogers Centre', 'altitude': 86, 'capacity': 53506, 'year_opened': 1989},
        3: {'name': 'Fenway Park', 'altitude': 21, 'capacity': 37755, 'year_opened': 1912},
        4: {'name': 'Oriole Park at Camden Yards', 'altitude': 20, 'capacity': 45971, 'year_opened': 1992},
        5: {'name': 'Tropicana Field', 'altitude': 6, 'capacity': 31042, 'year_opened': 1990},
        6: {'name': 'Comerica Park', 'altitude': 645, 'capacity': 41083, 'year_opened': 2000},
        7: {'name': 'Kauffman Stadium', 'altitude': 750, 'capacity': 37903, 'year_opened': 1973},
        8: {'name': 'Guaranteed Rate Field', 'altitude': 595, 'capacity': 40615, 'year_opened': 1991},
        9: {'name': 'Minute Maid Park', 'altitude': 20, 'capacity': 41168, 'year_opened': 2000},
        10: {'name': 'Globe Life Field', 'altitude': 551, 'capacity': 40300, 'year_opened': 2020},
        11: {'name': 'Oakland Coliseum', 'altitude': 42, 'capacity': 46847, 'year_opened': 1968},
        12: {'name': 'T-Mobile Park', 'altitude': 21, 'capacity': 47929, 'year_opened': 1999},
        13: {'name': 'Angel Stadium', 'altitude': 160, 'capacity': 45483, 'year_opened': 1966},
        14: {'name': 'Dodger Stadium', 'altitude': 340, 'capacity': 56000, 'year_opened': 1962},
        15: {'name': 'Petco Park', 'altitude': 20, 'capacity': 40209, 'year_opened': 2004},
        16: {'name': 'Chase Field', 'altitude': 1107, 'capacity': 48686, 'year_opened': 1998},
        17: {'name': 'Coors Field', 'altitude': 5277, 'capacity': 50430, 'year_opened': 1995},
        18: {'name': 'AT&T Park', 'altitude': 63, 'capacity': 41915, 'year_opened': 2000},
        19: {'name': 'Wrigley Field', 'altitude': 673, 'capacity': 41649, 'year_opened': 1914},
        20: {'name': 'Miller Park', 'altitude': 635, 'capacity': 41900, 'year_opened': 2001},
        21: {'name': 'Great American Ball Park', 'altitude': 540, 'capacity': 42271, 'year_opened': 2003},
        22: {'name': 'PNC Park', 'altitude': 730, 'capacity': 38365, 'year_opened': 2001},
        23: {'name': 'Citizens Bank Park', 'altitude': 40, 'capacity': 43647, 'year_opened': 2004},
        24: {'name': 'Nationals Park', 'altitude': 15, 'capacity': 41888, 'year_opened': 2008},
        25: {'name': 'Turner Field', 'altitude': 1050, 'capacity': 41800, 'year_opened': 1997},
        26: {'name': 'Marlins Park', 'altitude': 10, 'capacity': 36056, 'year_opened': 2012},
        27: {'name': 'Citi Field', 'altitude': 21, 'capacity': 41922, 'year_opened': 2009},
        28: {'name': 'Busch Stadium', 'altitude': 455, 'capacity': 46861, 'year_opened': 2006},
    }
    return stadiums.get(venue_id, {'name': 'Unknown', 'altitude': 0, 'capacity': 40000, 'year_opened': 2000})

# ============== API ENDPOINTS ==============

@app.route('/', methods=['GET'])
def index():
    """API health check and documentation"""
    return jsonify({
        'status': 'running',
        'message': 'MLB Prediction AI API',
        'version': '1.0.0',
        'endpoints': {
            'POST /predict': 'Predict winner and runs for a game',
            'GET /games/upcoming': 'Get upcoming games',
            'GET /teams': 'Get all MLB teams',
            'GET /health': 'API health status'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': ml_model.winner_model is not None
    }), 200

@app.route('/teams', methods=['GET'])
def get_teams():
    """Get all MLB teams"""
    try:
        teams = mlb_client.get_teams()
        return jsonify({
            'status': 'success',
            'total_teams': len(teams),
            'teams': teams.to_dict('records')
        }), 200
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/games/upcoming', methods=['GET'])
def get_upcoming_games():
    """Get upcoming games"""
    try:
        days = request.args.get('days', 7, type=int)
        games = mlb_client.get_upcoming_games(days_ahead=days)
        
        return jsonify({
            'status': 'success',
            'total_games': len(games),
            'games': games.to_dict('records')
        }), 200
    except Exception as e:
        logger.error(f"Error fetching upcoming games: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict_game():
    """
    Predict game winner and total runs
    
    Expected JSON:
    {
        "home_team_id": int,
        "away_team_id": int,
        "game_datetime": "2024-09-10T19:05:00Z",
        "venue_id": int
    }
    """
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['home_team_id', 'away_team_id', 'game_datetime', 'venue_id']
        if not all(field in data for field in required_fields):
            return jsonify({
                'status': 'error',
                'message': f'Missing required fields: {required_fields}'
            }), 400
        
        home_team_id = data['home_team_id']
        away_team_id = data['away_team_id']
        game_datetime = data['game_datetime']
        venue_id = data['venue_id']
        
        logger.info(f"Predicting game: Home={home_team_id}, Away={away_team_id}")
        
        # Get team stats
        home_stats = extract_stats_from_api(get_team_info(home_team_id))
        away_stats = extract_stats_from_api(get_team_info(away_team_id))
        
        # Get weather data (using stadium coordinates)
        stadium_info = get_stadium_info(venue_id)
        stadium_name = stadium_info.get('name', 'Unknown')
        
        from config import Config
        if stadium_name in Config.STADIUM_COORDS:
            lat, lon = Config.STADIUM_COORDS[stadium_name]
            weather = weather_client.get_forecast(lat, lon, game_datetime)
        else:
            weather = {
                'temperature': 70, 'humidity': 50, 'wind_speed': 0,
                'wind_direction': 0, 'pressure': 1013, 'rain': 0
            }
        
        # Process features
        game_features = data_processor.process_game_data(
            home_team=mlb_client.get_teams()[mlb_client.get_teams()['team_id'] == home_team_id]['name'].iloc[0] if len(mlb_client.get_teams()[mlb_client.get_teams()['team_id'] == home_team_id]) > 0 else 'Home Team',
            away_team=mlb_client.get_teams()[mlb_client.get_teams()['team_id'] == away_team_id]['name'].iloc[0] if len(mlb_client.get_teams()[mlb_client.get_teams()['team_id'] == away_team_id]) > 0 else 'Away Team',
            game_date=game_datetime,
            home_stats=home_stats,
            away_stats=away_stats,
            weather=weather,
            venue_info=stadium_info
        )
        
        # Make predictions
        if ml_model.winner_model is None:
            logger.warning("Models not trained yet, returning mock prediction")
            prediction = {
                'status': 'success',
                'message': 'Models not trained yet',
                'prediction': {
                    'winner': 'home',
                    'confidence': 55,
                    'note': 'This is a mock prediction. Train models with real data for accurate predictions.'
                }
            }
        else:
            winner_pred = ml_model.predict_winner(game_features)
            runs_pred = ml_model.predict_total_runs(game_features)
            
            prediction = {
                'status': 'success',
                'prediction': {
                    'winner': winner_pred,
                    'total_runs': runs_pred
                }
            }
        
        return jsonify(prediction), 200
        
    except Exception as e:
        logger.error(f"Error making prediction: {e}", exc_info=True)
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/models/status', methods=['GET'])
def model_status():
    """Get status of trained models"""
    return jsonify({
        'status': 'success',
        'models': {
            'winner_model': 'trained' if ml_model.winner_model is not None else 'not_trained',
            'runs_model': 'trained' if ml_model.runs_model is not None else 'not_trained'
        },
        'message': 'Use the training script to train models with historical data'
    }), 200

@app.route('/models/load', methods=['POST'])
def load_models():
    """Load pre-trained models"""
    try:
        from config import Config
        ml_model.load_models(
            Config.MODEL_WINNER_PATH,
            Config.MODEL_RUNS_PATH,
            Config.SCALER_PATH
        )
        return jsonify({'status': 'success', 'message': 'Models loaded successfully'}), 200
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

# ============== ERROR HANDLERS ==============

@app.errorhandler(404)
def not_found(error):
    return jsonify({'status': 'error', 'message': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'status': 'error', 'message': 'Internal server error'}), 500

# ============== MAIN ==============

if __name__ == '__main__':
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )
