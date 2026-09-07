"""
Machine Learning Models for MLB Predictions
Includes models for predicting game winners and total runs
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, r2_score
import xgboost as xgb
import joblib
import logging
from typing import Tuple, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLBPredictionModel:
    """Master class for MLB prediction models"""
    
    def __init__(self):
        self.winner_model = None
        self.runs_model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.winner_classes = ['home_win', 'away_win']
    
    # ============== WINNER PREDICTION MODEL ==============
    
    def train_winner_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train model to predict game winner (home win vs away win)
        
        Args:
            X: Feature matrix (features)
            y: Target vector (0 = away win, 1 = home win)
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with model metrics
        """
        logger.info("Training winner prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost classifier (best for binary classification)
        self.winner_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.winner_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Predictions
        y_pred = self.winner_model.predict(X_test_scaled)
        y_pred_proba = self.winner_model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'auc_score': self._calculate_auc(y_test, y_pred_proba[:, 1])
        }
        
        logger.info(f"Winner Model - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
        
        self.feature_columns = X.columns.tolist()
        
        return metrics
    
    # ============== RUNS PREDICTION MODEL ==============
    
    def train_runs_model(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train model to predict total runs in game
        
        Args:
            X: Feature matrix
            y: Target vector (total runs)
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with model metrics
        """
        logger.info("Training runs prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train XGBoost regressor (best for regression)
        self.runs_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        self.runs_model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=20,
            verbose=False
        )
        
        # Predictions
        y_pred = self.runs_model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_test - y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2
        }
        
        logger.info(f"Runs Model - RMSE: {rmse:.4f}, R²: {r2:.4f}, MAE: {mae:.4f}")
        
        return metrics
    
    # ============== PREDICTION METHODS ==============
    
    def predict_winner(self, game_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict game winner
        
        Args:
            game_features: Single game feature row
        
        Returns:
            Dictionary with prediction and probability
        """
        if self.winner_model is None:
            raise ValueError("Winner model not trained yet")
        
        features_scaled = self.scaler.transform(game_features)
        prediction = self.winner_model.predict(features_scaled)[0]
        probabilities = self.winner_model.predict_proba(features_scaled)[0]
        
        winner = self.winner_classes[prediction]
        confidence = max(probabilities) * 100
        
        return {
            'prediction': winner,
            'confidence': confidence,
            'probabilities': {
                'away_win': probabilities[0] * 100,
                'home_win': probabilities[1] * 100
            }
        }
    
    def predict_total_runs(self, game_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict total runs in game
        
        Args:
            game_features: Single game feature row
        
        Returns:
            Dictionary with predicted runs and range
        """
        if self.runs_model is None:
            raise ValueError("Runs model not trained yet")
        
        features_scaled = self.scaler.transform(game_features)
        predicted_runs = self.runs_model.predict(features_scaled)[0]
        
        # Estimate confidence interval (±1.96 * std)
        predicted_runs = max(0, predicted_runs)  # Can't have negative runs
        lower_bound = max(0, predicted_runs - 2.5)
        upper_bound = predicted_runs + 2.5
        
        return {
            'predicted_runs': round(predicted_runs, 1),
            'lower_bound': round(lower_bound, 1),
            'upper_bound': round(upper_bound, 1),
            'over_under_8': 'OVER' if predicted_runs > 8 else 'UNDER'
        }
    
    def predict_game(self, game_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete game prediction (winner + runs)
        
        Args:
            game_features: Single game feature row
        
        Returns:
            Dictionary with complete prediction
        """
        winner_pred = self.predict_winner(game_features)
        runs_pred = self.predict_total_runs(game_features)
        
        return {
            'winner_prediction': winner_pred,
            'runs_prediction': runs_pred,
            'combined_analysis': {
                'likely_outcome': f"{winner_pred['prediction'].upper()} with {runs_pred['predicted_runs']} total runs",
                'confidence_level': 'HIGH' if winner_pred['confidence'] > 65 else 'MEDIUM' if winner_pred['confidence'] > 55 else 'LOW'
            }
        }
    
    # ============== MODEL PERSISTENCE ==============
    
    def save_models(self, winner_path: str, runs_path: str, scaler_path: str):
        """Save trained models to disk"""
        if self.winner_model:
            joblib.dump(self.winner_model, winner_path)
            logger.info(f"Winner model saved to {winner_path}")
        
        if self.runs_model:
            joblib.dump(self.runs_model, runs_path)
            logger.info(f"Runs model saved to {runs_path}")
        
        if self.scaler:
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
    
    def load_models(self, winner_path: str, runs_path: str, scaler_path: str):
        """Load trained models from disk"""
        try:
            self.winner_model = joblib.load(winner_path)
            self.runs_model = joblib.load(runs_path)
            self.scaler = joblib.load(scaler_path)
            logger.info("Models loaded successfully")
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise
    
    # ============== UTILITY METHODS ==============
    
    def get_feature_importance(self, model_type: str = 'winner') -> pd.DataFrame:
        """Get feature importance from trained models"""
        model = self.winner_model if model_type == 'winner' else self.runs_model
        
        if model is None:
            raise ValueError(f"{model_type} model not trained yet")
        
        importances = model.feature_importances_
        features_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        return features_df
    
    def _calculate_auc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
        """Calculate AUC score"""
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, y_pred_proba)
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, model_type: str = 'winner', cv: int = 5) -> Dict[str, float]:
        """Perform cross-validation"""
        model = self.winner_model if model_type == 'winner' else self.runs_model
        
        if model is None:
            raise ValueError(f"{model_type} model not trained yet")
        
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(model, X_scaled, y, cv=cv)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores.tolist()
        }
