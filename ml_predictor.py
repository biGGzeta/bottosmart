"""
Machine Learning predictor for market movements
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import logging
from typing import Dict, List, Optional, Tuple
import pickle
import os
from datetime import datetime, timedelta

from market_analyzer import OpportunityType

class MLPredictor:
    """
    Machine learning-based market movement predictor
    Enhanced for opportunistic trading strategies
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.model_performance = {}
        
        # Model parameters for opportunistic trading
        self.rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.gb_params = {
            'n_estimators': 150,
            'max_depth': 8,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        
        # Load existing models if available
        self.load_models()
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive features for ML prediction
        Focused on opportunistic trading signals
        """
        features = df.copy()
        close = features['close']
        high = features['high']
        low = features['low']
        volume = features['volume'] if 'volume' in features.columns else pd.Series(index=features.index, data=1)
        
        # Price-based features
        features['returns'] = close.pct_change()
        features['log_returns'] = np.log(close / close.shift(1))
        features['volatility'] = features['returns'].rolling(14).std()
        features['momentum'] = close / close.shift(10) - 1
        
        # Technical indicators
        features['sma_5'] = close.rolling(5).mean()
        features['sma_10'] = close.rolling(10).mean()
        features['sma_20'] = close.rolling(20).mean()
        features['ema_12'] = close.ewm(span=12).mean()
        features['ema_26'] = close.ewm(span=26).mean()
        
        # Price position relative to moving averages
        features['price_vs_sma5'] = close / features['sma_5'] - 1
        features['price_vs_sma10'] = close / features['sma_10'] - 1
        features['price_vs_sma20'] = close / features['sma_20'] - 1
        
        # Bollinger Bands
        bb_window = 20
        bb_std = close.rolling(bb_window).std()
        bb_mean = close.rolling(bb_window).mean()
        features['bb_upper'] = bb_mean + (2 * bb_std)
        features['bb_lower'] = bb_mean - (2 * bb_std)
        features['bb_position'] = (close - bb_lower) / (features['bb_upper'] - bb_lower)
        features['bb_squeeze'] = bb_std / bb_mean  # Volatility squeeze indicator
        
        # RSI and momentum
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        features['rsi'] = 100 - (100 / (1 + rs))
        features['rsi_divergence'] = features['rsi'] - features['rsi'].rolling(5).mean()
        
        # MACD
        macd_line = features['ema_12'] - features['ema_26']
        macd_signal = macd_line.ewm(span=9).mean()
        features['macd'] = macd_line
        features['macd_signal'] = macd_signal
        features['macd_histogram'] = macd_line - macd_signal
        features['macd_momentum'] = features['macd_histogram'].diff()
        
        # Volume features (if available)
        if 'volume' in df.columns:
            features['volume_sma'] = volume.rolling(20).mean()
            features['volume_ratio'] = volume / features['volume_sma']
            features['price_volume'] = close * volume
            features['vwap'] = (features['price_volume'].rolling(20).sum() / volume.rolling(20).sum())
            features['price_vs_vwap'] = close / features['vwap'] - 1
        else:
            # Create dummy volume features
            features['volume_ratio'] = 1.0
            features['price_vs_vwap'] = 0.0
            
        # Candlestick patterns
        features['doji'] = abs(close - df['open']) / (high - low)
        features['upper_shadow'] = (high - np.maximum(close, df['open'])) / (high - low)
        features['lower_shadow'] = (np.minimum(close, df['open']) - low) / (high - low)
        features['body_size'] = abs(close - df['open']) / (high - low)
        
        # Market structure features
        features['higher_high'] = (high > high.shift(1)).astype(int)
        features['lower_low'] = (low < low.shift(1)).astype(int)
        features['inside_bar'] = ((high < high.shift(1)) & (low > low.shift(1))).astype(int)
        features['outside_bar'] = ((high > high.shift(1)) & (low < low.shift(1))).astype(int)
        
        # Support and resistance levels
        features['resistance_break'] = (close > high.rolling(20).max().shift(1)).astype(int)
        features['support_break'] = (close < low.rolling(20).min().shift(1)).astype(int)
        
        # Time-based features
        features['hour'] = pd.to_datetime(features.index).hour if hasattr(features.index, 'hour') else 0
        features['day_of_week'] = pd.to_datetime(features.index).dayofweek if hasattr(features.index, 'dayofweek') else 0
        
        # Lag features (previous periods)
        for lag in [1, 2, 3, 5]:
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            features[f'rsi_lag_{lag}'] = features['rsi'].shift(lag)
            features[f'volume_ratio_lag_{lag}'] = features['volume_ratio'].shift(lag)
        
        # Forward-looking features for targets
        features['future_return_1'] = features['returns'].shift(-1)  # Next period return
        features['future_return_5'] = close.shift(-5) / close - 1   # 5-period return
        
        # Clean and select features
        feature_cols = [col for col in features.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
        self.feature_columns = feature_cols
        
        return features[feature_cols].dropna()
    
    def create_targets(self, df: pd.DataFrame, opportunity_type: OpportunityType) -> np.ndarray:
        """
        Create prediction targets based on opportunity type
        """
        if 'future_return_1' not in df.columns:
            return np.array([])
            
        future_returns = df['future_return_1'].values
        
        if opportunity_type == OpportunityType.MOMENTUM_BREAKOUT:
            # For momentum, predict strong upward moves (>2%)
            targets = (future_returns > 0.02).astype(int)
        elif opportunity_type == OpportunityType.REVERSAL_BOUNCE:
            # For reversals, predict bounce after decline
            targets = (future_returns > 0.015).astype(int)  # Lower threshold
        elif opportunity_type == OpportunityType.VOLATILITY_EXPANSION:
            # For volatility, predict significant moves in either direction
            targets = (np.abs(future_returns) > 0.025).astype(int)
        else:  # TREND_CONTINUATION
            # For trend continuation, predict sustained moves
            targets = (future_returns > 0.01).astype(int)
            
        return targets
    
    def train_models(self, df: pd.DataFrame, opportunity_type: OpportunityType):
        """
        Train ML models for specific opportunity type
        """
        self.logger.info(f"Training models for {opportunity_type.value}")
        
        # Prepare features
        features_df = self.prepare_features(df)
        if len(features_df) < 100:  # Need minimum data
            self.logger.warning(f"Insufficient data for training {opportunity_type.value}")
            return
            
        # Create targets
        targets = self.create_targets(features_df, opportunity_type)
        if len(targets) == 0:
            return
            
        # Align features and targets
        min_len = min(len(features_df), len(targets))
        X = features_df.iloc[:min_len].values
        y = targets[:min_len]
        
        # Remove any remaining NaN values
        mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
        X = X[mask]
        y = y[mask]
        
        if len(X) < 50:
            self.logger.warning(f"Insufficient clean data for training {opportunity_type.value}")
            return
            
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train models
        models = {
            'random_forest': RandomForestClassifier(**self.rf_params),
            'gradient_boosting': GradientBoostingClassifier(**self.gb_params)
        }
        
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            try:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                score = accuracy_score(y_test, y_pred)
                
                self.logger.info(f"{name} accuracy for {opportunity_type.value}: {score:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                self.logger.error(f"Error training {name} for {opportunity_type.value}: {e}")
                
        # Store best model
        if best_model is not None and best_score > 0.52:  # Only store if better than random
            self.models[opportunity_type.value] = best_model
            self.scalers[opportunity_type.value] = scaler
            self.model_performance[opportunity_type.value] = best_score
            
            # Save model
            self.save_models()
            
            self.logger.info(f"✅ Trained model for {opportunity_type.value} with accuracy {best_score:.3f}")
        else:
            self.logger.warning(f"❌ Model performance too low for {opportunity_type.value}: {best_score:.3f}")
    
    async def predict_price_movement(self, df: pd.DataFrame, opportunity_type: OpportunityType) -> float:
        """
        Predict price movement probability for given opportunity type
        """
        try:
            if opportunity_type.value not in self.models:
                # Train model if not exists
                if len(df) >= 200:  # Need sufficient data for training
                    self.train_models(df, opportunity_type)
                
                if opportunity_type.value not in self.models:
                    return 0.5  # Return neutral confidence if no model
            
            # Prepare features for prediction
            features_df = self.prepare_features(df)
            if len(features_df) == 0:
                return 0.5
                
            # Get latest features
            latest_features = features_df.iloc[-1:].values
            
            # Handle NaN values
            if np.isnan(latest_features).any():
                return 0.5
                
            # Scale features
            scaler = self.scalers[opportunity_type.value]
            latest_features_scaled = scaler.transform(latest_features)
            
            # Make prediction
            model = self.models[opportunity_type.value]
            prediction_proba = model.predict_proba(latest_features_scaled)[0]
            
            # Return probability of positive outcome
            confidence = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
            
            # Adjust confidence based on model performance
            model_performance = self.model_performance.get(opportunity_type.value, 0.5)
            adjusted_confidence = confidence * model_performance + (1 - model_performance) * 0.5
            
            return float(adjusted_confidence)
            
        except Exception as e:
            self.logger.error(f"Error in prediction for {opportunity_type.value}: {e}")
            return 0.5
    
    def save_models(self):
        """Save trained models to disk"""
        try:
            model_dir = 'models'
            os.makedirs(model_dir, exist_ok=True)
            
            # Save models
            for name, model in self.models.items():
                model_path = os.path.join(model_dir, f'{name}_model.pkl')
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
            # Save scalers
            for name, scaler in self.scalers.items():
                scaler_path = os.path.join(model_dir, f'{name}_scaler.pkl')
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
                    
            # Save feature columns
            with open(os.path.join(model_dir, 'feature_columns.pkl'), 'wb') as f:
                pickle.dump(self.feature_columns, f)
                
            # Save performance metrics
            with open(os.path.join(model_dir, 'performance.pkl'), 'wb') as f:
                pickle.dump(self.model_performance, f)
                
        except Exception as e:
            self.logger.error(f"Error saving models: {e}")
    
    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            model_dir = 'models'
            if not os.path.exists(model_dir):
                return
                
            # Load feature columns
            feature_path = os.path.join(model_dir, 'feature_columns.pkl')
            if os.path.exists(feature_path):
                with open(feature_path, 'rb') as f:
                    self.feature_columns = pickle.load(f)
                    
            # Load performance metrics
            perf_path = os.path.join(model_dir, 'performance.pkl')
            if os.path.exists(perf_path):
                with open(perf_path, 'rb') as f:
                    self.model_performance = pickle.load(f)
            
            # Load models and scalers
            for opportunity_type in OpportunityType:
                model_path = os.path.join(model_dir, f'{opportunity_type.value}_model.pkl')
                scaler_path = os.path.join(model_dir, f'{opportunity_type.value}_scaler.pkl')
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        self.models[opportunity_type.value] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[opportunity_type.value] = pickle.load(f)
                        
                    self.logger.info(f"✅ Loaded model for {opportunity_type.value}")
                    
        except Exception as e:
            self.logger.error(f"Error loading models: {e}")
    
    def get_model_info(self) -> Dict:
        """Get information about loaded models"""
        return {
            'loaded_models': list(self.models.keys()),
            'model_performance': self.model_performance,
            'feature_count': len(self.feature_columns),
            'last_updated': datetime.now().isoformat()
        }
    
    def retrain_if_needed(self, df: pd.DataFrame):
        """
        Retrain models if performance degrades or new data available
        """
        # Check if we have enough new data
        if len(df) < 500:  # Need substantial data for retraining
            return
            
        # Retrain models periodically or if performance is low
        for opportunity_type in OpportunityType:
            performance = self.model_performance.get(opportunity_type.value, 0.0)
            
            # Retrain if performance is low or model doesn't exist
            if performance < 0.55 or opportunity_type.value not in self.models:
                self.logger.info(f"Retraining model for {opportunity_type.value} (performance: {performance:.3f})")
                self.train_models(df, opportunity_type)