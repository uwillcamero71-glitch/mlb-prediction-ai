import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration"""
    MLB_API_BASE_URL = os.getenv('MLB_API_BASE_URL', 'https://statsapi.mlb.com/api/v1')
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY', '')
    WEATHER_API_URL = os.getenv('WEATHER_API_URL', 'https://api.openweathermap.org/data/2.5')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///mlb_predictions.db')
    
    # Model paths
    MODEL_WINNER_PATH = 'models/winner_model.pkl'
    MODEL_RUNS_PATH = 'models/runs_model.pkl'
    SCALER_PATH = 'models/scaler.pkl'
    
    # Stadium coordinates for weather (lat, lon)
    STADIUM_COORDS = {
        'Yankee Stadium': (40.8296, -73.9262),
        'Fenway Park': (42.3461, -71.0972),
        'Wrigley Field': (41.9484, -87.6553),
        'Dodger Stadium': (34.0739, -118.2400),
        'Coors Field': (39.7560, -104.9844),
        'AT&T Park': (37.7787, -122.3893),
        'Great American Ball Park': (39.0979, -84.5061),
        'Comerica Park': (42.3391, -83.0485),
        'Minute Maid Park': (29.7571, -95.3555),
        'Tropicana Field': (27.7683, -82.6539),
        'Kauffman Stadium': (39.0521, -94.4803),
        'Globe Life Field': (32.7555, -97.0833),
        'Fenway Park': (42.3461, -71.0972),
        'Busch Stadium': (38.6226, -90.1928),
        'Miller Park': (43.0285, -87.9711),
        'PNC Park': (40.4406, -80.0027),
        'Citizens Bank Park': (39.9026, -75.1675),
        'Chase Field': (33.4454, -112.0674),
        'Petco Park': (32.7075, -117.1571),
        'Citi Field': (40.7571, -73.8458),
        'Turner Field': (33.7335, -84.3900),
        'Marlins Park': (25.7282, -80.2300),
        'Nationals Park': (38.8731, -77.0369),
        'Guaranteed Rate Field': (41.8299, -87.6338),
        'Target Field': (44.9819, -93.2775),
        'Oakland Coliseum': (37.7213, -122.2003),
        'Safeco Field': (47.5911, -122.3327),
        'Angel Stadium': (33.7457, -117.8725),
        'Rogers Centre': (43.6426, -79.3957),
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    DATABASE_URL = 'sqlite:///:memory:'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
