"""
MLB Stats API Client
Fetches real-time data from MLB's official API
"""
import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLBAPIClient:
    """Client for MLB Stats API"""
    
    def __init__(self, base_url: str = 'https://statsapi.mlb.com/api/v1'):
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_teams(self) -> pd.DataFrame:
        """Get all MLB teams with their info"""
        try:
            url = f"{self.base_url}/teams"
            response = self.session.get(url, params={'sportId': 1})
            response.raise_for_status()
            
            teams_data = []
            for team in response.json()['teams']:
                teams_data.append({
                    'team_id': team['id'],
                    'name': team['name'],
                    'code': team['teamCode'],
                    'division': team['division']['name'],
                    'league': team['league']['name'],
                    'venue_id': team.get('venue', {}).get('id'),
                    'venue_name': team.get('venue', {}).get('name'),
                })
            
            return pd.DataFrame(teams_data)
        except Exception as e:
            logger.error(f"Error fetching teams: {e}")
            return pd.DataFrame()
    
    def get_team_stats(self, team_id: int, season: int = None) -> Dict:
        """Get team statistics for a season"""
        if season is None:
            season = datetime.now().year
        
        try:
            url = f"{self.base_url}/teams/{team_id}"
            params = {'sportId': 1}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            team_data = response.json()['teams'][0]
            
            return {
                'team_id': team_id,
                'name': team_data['name'],
                'season': season,
                'record': team_data.get('record', {}),
                'runs_scored': team_data.get('recordSource', {}).get('runsScored'),
                'runs_allowed': team_data.get('recordSource', {}).get('runsAllowed'),
            }
        except Exception as e:
            logger.error(f"Error fetching team stats for team {team_id}: {e}")
            return {}
    
    def get_season_stats(self, team_id: int, season: int = None) -> pd.DataFrame:
        """Get detailed season stats for a team"""
        if season is None:
            season = datetime.now().year
        
        try:
            url = f"{self.base_url}/teams/{team_id}/stats"
            params = {
                'group': 'hitting,pitching,fielding',
                'type': 'season',
                'sportId': 1
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            stats = response.json()['stats']
            
            stats_dict = {}
            for stat_group in stats:
                group_name = stat_group['group']['displayName'].lower()
                for record in stat_group['records']:
                    stats_dict[group_name] = record['stats']
            
            return stats_dict
        except Exception as e:
            logger.error(f"Error fetching season stats: {e}")
            return {}
    
    def get_games_by_date(self, date: str) -> pd.DataFrame:
        """Get all games for a specific date (YYYY-MM-DD format)"""
        try:
            url = f"{self.base_url}/schedule"
            params = {
                'sportId': 1,
                'date': date
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            games_data = []
            for game in response.json()['dates'][0].get('games', []):
                games_data.append({
                    'game_id': game['gamePk'],
                    'date': game['gameDateTime'],
                    'home_team': game['teams']['home']['team']['name'],
                    'home_team_id': game['teams']['home']['team']['id'],
                    'away_team': game['teams']['away']['team']['name'],
                    'away_team_id': game['teams']['away']['team']['id'],
                    'status': game['status']['abstractGameState'],
                    'venue': game.get('venue', {}).get('name'),
                    'venue_id': game.get('venue', {}).get('id'),
                })
            
            return pd.DataFrame(games_data)
        except Exception as e:
            logger.error(f"Error fetching games for {date}: {e}")
            return pd.DataFrame()
    
    def get_game_details(self, game_id: int) -> Dict:
        """Get detailed information about a specific game"""
        try:
            url = f"{self.base_url}/game/{game_id}/boxscore"
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            
            home_team = data['teams']['home']
            away_team = data['teams']['away']
            
            return {
                'game_id': game_id,
                'home_team': home_team['team']['name'],
                'home_team_id': home_team['team']['id'],
                'home_runs': home_team['teamStats']['batting'].get('runs', 0),
                'home_hits': home_team['teamStats']['batting'].get('hits', 0),
                'home_errors': home_team['teamStats']['fielding'].get('errors', 0),
                'away_team': away_team['team']['name'],
                'away_team_id': away_team['team']['id'],
                'away_runs': away_team['teamStats']['batting'].get('runs', 0),
                'away_hits': away_team['teamStats']['batting'].get('hits', 0),
                'away_errors': away_team['teamStats']['fielding'].get('errors', 0),
                'venue': data.get('venue', {}).get('name'),
                'winning_team': home_team['team']['name'] if home_team['teamStats']['batting']['runs'] > away_team['teamStats']['batting']['runs'] else away_team['team']['name'],
            }
        except Exception as e:
            logger.error(f"Error fetching game details for {game_id}: {e}")
            return {}
    
    def get_player_stat_data(self, team_id: int, stat_type: str = 'hitting') -> pd.DataFrame:
        """Get player-level statistics for a team"""
        try:
            url = f"{self.base_url}/teams/{team_id}/roster"
            response = self.session.get(url)
            response.raise_for_status()
            
            roster = response.json()['roster']
            
            player_data = []
            for player in roster:
                player_data.append({
                    'player_id': player['personId'],
                    'name': player['person']['fullName'],
                    'position': player['position']['name'],
                    'number': player['jerseyNumber'],
                })
            
            return pd.DataFrame(player_data)
        except Exception as e:
            logger.error(f"Error fetching player data: {e}")
            return pd.DataFrame()
    
    def get_upcoming_games(self, days_ahead: int = 7) -> pd.DataFrame:
        """Get upcoming games for the next N days"""
        games_list = []
        
        for i in range(days_ahead):
            date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            games = self.get_games_by_date(date)
            if not games.empty:
                games_list.append(games)
        
        return pd.concat(games_list, ignore_index=True) if games_list else pd.DataFrame()
    
    def get_pitcher_stats(self, pitcher_id: int) -> Dict:
        """Get pitcher statistics"""
        try:
            url = f"{self.base_url}/people/{pitcher_id}"
            params = {'hydrate': 'stats'}
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            pitcher = response.json()['people'][0]
            
            return {
                'pitcher_id': pitcher_id,
                'name': pitcher['fullName'],
                'age': pitcher.get('currentAge'),
                'throws': pitcher.get('pitchHand', {}).get('description'),
                'stats': pitcher.get('stats', [])
            }
        except Exception as e:
            logger.error(f"Error fetching pitcher stats: {e}")
            return {}

class WeatherClient:
    """Client for weather data"""
    
    def __init__(self, api_key: str, base_url: str = 'https://api.openweathermap.org/data/2.5'):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()
    
    def get_weather(self, lat: float, lon: float) -> Dict:
        """Get current weather for stadium coordinates"""
        try:
            url = f"{self.base_url}/weather"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            return {
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'wind_speed': data['wind'].get('speed', 0),
                'wind_direction': data['wind'].get('deg', 0),
                'clouds': data['clouds']['all'],
                'rain': data.get('rain', {}).get('1h', 0),
                'description': data['weather'][0]['description'],
            }
        except Exception as e:
            logger.error(f"Error fetching weather: {e}")
            return {}
    
    def get_forecast(self, lat: float, lon: float, game_datetime: str) -> Dict:
        """Get weather forecast for a specific game time"""
        try:
            url = f"{self.base_url}/forecast"
            params = {
                'lat': lat,
                'lon': lon,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Find the closest forecast to game time
            game_time = datetime.fromisoformat(game_datetime.replace('Z', '+00:00'))
            closest_forecast = None
            min_diff = timedelta.max
            
            for forecast in data['list']:
                forecast_time = datetime.fromtimestamp(forecast['dt'])
                diff = abs((forecast_time - game_time).total_seconds())
                
                if diff < min_diff.total_seconds():
                    min_diff = timedelta(seconds=diff)
                    closest_forecast = forecast
            
            if closest_forecast:
                return {
                    'temperature': closest_forecast['main']['temp'],
                    'humidity': closest_forecast['main']['humidity'],
                    'wind_speed': closest_forecast['wind'].get('speed', 0),
                    'wind_direction': closest_forecast['wind'].get('deg', 0),
                    'clouds': closest_forecast['clouds']['all'],
                    'rain': closest_forecast.get('rain', {}).get('3h', 0),
                    'description': closest_forecast['weather'][0]['description'],
                }
            
            return {}
        except Exception as e:
            logger.error(f"Error fetching forecast: {e}")
            return {}
