"""
Configuration module for BottoSmart trading bot
"""
import os
from typing import Dict, List, Optional
from pydantic import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class TradingConfig(BaseSettings):
    """Trading configuration settings"""
    
    # Exchange settings
    EXCHANGE: str = os.getenv("EXCHANGE", "binance")
    API_KEY: str = os.getenv("API_KEY", "")
    API_SECRET: str = os.getenv("API_SECRET", "")
    SANDBOX: bool = os.getenv("SANDBOX", "true").lower() == "true"
    
    # Trading pairs
    TRADING_PAIRS: List[str] = ["BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT", "SOL/USDT"]
    
    # Opportunistic strategy parameters
    MAX_POSITION_SIZE: float = 0.15  # 15% of portfolio per trade (aggressive)
    MIN_POSITION_SIZE: float = 0.02  # 2% minimum
    RISK_REWARD_RATIO: float = 1.5   # Minimum 1.5:1 reward/risk
    
    # Adaptive parameters
    VOLATILITY_THRESHOLD: float = 0.02  # 2% volatility threshold
    MOMENTUM_THRESHOLD: float = 0.03    # 3% momentum threshold
    RSI_OVERBOUGHT: float = 75          # Aggressive RSI levels
    RSI_OVERSOLD: float = 25
    
    # Machine learning parameters
    PREDICTION_CONFIDENCE_THRESHOLD: float = 0.65
    LOOKBACK_PERIOD: int = 100
    FEATURE_WINDOW: int = 14
    
    # Risk management
    MAX_DAILY_LOSS: float = 0.05   # 5% max daily loss
    MAX_DRAWDOWN: float = 0.15     # 15% max drawdown
    STOP_LOSS: float = 0.025       # 2.5% stop loss (tight for opportunistic)
    TAKE_PROFIT: float = 0.06      # 6% take profit (aggressive target)
    
    # Portfolio management
    INITIAL_BALANCE: float = 10000.0
    LEVERAGE: float = 1.0  # Start with no leverage
    
    class Config:
        env_file = ".env"

# Global configuration instance
config = TradingConfig()