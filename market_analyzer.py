"""
Market data analysis module for opportunistic trading
"""
import numpy as np
import pandas as pd
import talib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class MarketTrend(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"

class OpportunityType(Enum):
    MOMENTUM_BREAKOUT = "momentum_breakout"
    REVERSAL_BOUNCE = "reversal_bounce"
    VOLATILITY_EXPANSION = "volatility_expansion"
    TREND_CONTINUATION = "trend_continuation"

@dataclass
class MarketOpportunity:
    """Represents a trading opportunity"""
    pair: str
    opportunity_type: OpportunityType
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    expected_return: float
    risk_score: float
    timestamp: pd.Timestamp

class MarketAnalyzer:
    """Advanced market analysis for opportunistic trading"""
    
    def __init__(self):
        self.indicators = {}
        
    def analyze_market_data(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        Comprehensive market analysis for opportunistic trading
        """
        if len(df) < 50:
            return {"trend": MarketTrend.NEUTRAL, "opportunities": []}
            
        # Calculate technical indicators
        indicators = self._calculate_indicators(df)
        
        # Analyze market structure
        market_structure = self._analyze_market_structure(df, indicators)
        
        # Detect opportunities
        opportunities = self._detect_opportunities(df, indicators, market_structure, pair)
        
        # Assess overall market sentiment
        sentiment = self._calculate_market_sentiment(indicators)
        
        return {
            "trend": market_structure["trend"],
            "volatility": market_structure["volatility"],
            "momentum": market_structure["momentum"],
            "sentiment": sentiment,
            "opportunities": opportunities,
            "indicators": indicators
        }
    
    def _calculate_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive technical indicators"""
        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values if 'volume' in df.columns else None
        
        indicators = {
            # Trend indicators
            'sma_20': talib.SMA(close, timeperiod=20),
            'ema_12': talib.EMA(close, timeperiod=12),
            'ema_26': talib.EMA(close, timeperiod=26),
            'macd': talib.MACD(close)[0],
            'macd_signal': talib.MACD(close)[1],
            'macd_hist': talib.MACD(close)[2],
            
            # Momentum indicators
            'rsi': talib.RSI(close, timeperiod=14),
            'rsi_fast': talib.RSI(close, timeperiod=7),  # For quick signals
            'stoch_k': talib.STOCH(high, low, close)[0],
            'williams_r': talib.WILLR(high, low, close),
            
            # Volatility indicators
            'bollinger_upper': talib.BBANDS(close)[0],
            'bollinger_middle': talib.BBANDS(close)[1],
            'bollinger_lower': talib.BBANDS(close)[2],
            'atr': talib.ATR(high, low, close, timeperiod=14),
            
            # Support/Resistance
            'pivot_high': self._find_pivot_points(high, 'high'),
            'pivot_low': self._find_pivot_points(low, 'low'),
            
            # Price action
            'price_change': np.diff(close, prepend=close[0]) / close * 100,
            'volatility': pd.Series(close).rolling(20).std() / pd.Series(close).rolling(20).mean() * 100
        }
        
        if volume is not None:
            indicators.update({
                'volume_sma': talib.SMA(volume.astype(float), timeperiod=20),
                'obv': talib.OBV(close, volume.astype(float)),
                'ad_line': talib.AD(high, low, close, volume.astype(float))
            })
            
        return indicators
    
    def _find_pivot_points(self, data: np.ndarray, type_: str, window: int = 5) -> np.ndarray:
        """Find pivot highs and lows"""
        pivots = np.zeros_like(data)
        for i in range(window, len(data) - window):
            if type_ == 'high':
                if all(data[i] >= data[i-j] for j in range(1, window+1)) and \
                   all(data[i] >= data[i+j] for j in range(1, window+1)):
                    pivots[i] = data[i]
            else:  # low
                if all(data[i] <= data[i-j] for j in range(1, window+1)) and \
                   all(data[i] <= data[i+j] for j in range(1, window+1)):
                    pivots[i] = data[i]
        return pivots
    
    def _analyze_market_structure(self, df: pd.DataFrame, indicators: Dict) -> Dict:
        """Analyze overall market structure and trend"""
        close = df['close'].values
        current_price = close[-1]
        
        # Trend analysis (aggressive approach)
        ema_12 = indicators['ema_12'][-1]
        ema_26 = indicators['ema_26'][-1]
        sma_20 = indicators['sma_20'][-1]
        
        # Price relative to EMAs
        price_above_ema12 = current_price > ema_12
        price_above_ema26 = current_price > ema_26
        price_above_sma20 = current_price > sma_20
        ema_bullish = ema_12 > ema_26
        
        # MACD momentum
        macd = indicators['macd'][-1]
        macd_signal = indicators['macd_signal'][-1]
        macd_bullish = macd > macd_signal
        
        # Determine trend strength (opportunistic approach)
        bullish_signals = sum([price_above_ema12, price_above_ema26, price_above_sma20, ema_bullish, macd_bullish])
        
        if bullish_signals >= 4:
            trend = MarketTrend.STRONG_BULLISH
        elif bullish_signals >= 3:
            trend = MarketTrend.BULLISH
        elif bullish_signals >= 2:
            trend = MarketTrend.NEUTRAL
        elif bullish_signals >= 1:
            trend = MarketTrend.BEARISH
        else:
            trend = MarketTrend.STRONG_BEARISH
            
        # Volatility analysis
        current_volatility = indicators['volatility'][-1] if not np.isnan(indicators['volatility'][-1]) else 0
        avg_volatility = np.nanmean(indicators['volatility'][-20:])
        volatility_expansion = current_volatility > avg_volatility * 1.5
        
        # Momentum analysis
        price_momentum = (current_price - close[-20]) / close[-20] * 100 if len(close) >= 20 else 0
        
        return {
            "trend": trend,
            "volatility": current_volatility,
            "volatility_expansion": volatility_expansion,
            "momentum": price_momentum,
            "trend_strength": bullish_signals / 5.0
        }
    
    def _calculate_market_sentiment(self, indicators: Dict) -> float:
        """Calculate overall market sentiment score (0-1)"""
        rsi = indicators['rsi'][-1]
        stoch_k = indicators['stoch_k'][-1]
        williams_r = indicators['williams_r'][-1]
        
        # Normalize indicators to 0-1 scale
        rsi_normalized = rsi / 100
        stoch_normalized = stoch_k / 100
        williams_normalized = (williams_r + 100) / 100  # Williams %R is typically -100 to 0
        
        # Weighted sentiment (aggressive interpretation)
        sentiment = (rsi_normalized * 0.4 + stoch_normalized * 0.3 + williams_normalized * 0.3)
        return sentiment
    
    def _detect_opportunities(self, df: pd.DataFrame, indicators: Dict, 
                            market_structure: Dict, pair: str) -> List[MarketOpportunity]:
        """Detect aggressive trading opportunities"""
        opportunities = []
        close = df['close'].values
        current_price = close[-1]
        atr = indicators['atr'][-1]
        
        # Momentum breakout opportunities
        momentum_opp = self._detect_momentum_breakout(df, indicators, current_price, atr, pair)
        if momentum_opp:
            opportunities.append(momentum_opp)
            
        # Reversal bounce opportunities
        reversal_opp = self._detect_reversal_opportunity(df, indicators, current_price, atr, pair)
        if reversal_opp:
            opportunities.append(reversal_opp)
            
        # Volatility expansion opportunities
        volatility_opp = self._detect_volatility_opportunity(df, indicators, market_structure, current_price, atr, pair)
        if volatility_opp:
            opportunities.append(volatility_opp)
            
        return opportunities
    
    def _detect_momentum_breakout(self, df: pd.DataFrame, indicators: Dict, 
                                 current_price: float, atr: float, pair: str) -> Optional[MarketOpportunity]:
        """Detect aggressive momentum breakout opportunities"""
        rsi = indicators['rsi'][-1]
        rsi_fast = indicators['rsi_fast'][-1]
        macd_hist = indicators['macd_hist'][-1]
        bollinger_upper = indicators['bollinger_upper'][-1]
        
        # Aggressive momentum conditions
        strong_momentum = (
            rsi_fast > 60 and  # Fast RSI showing strength
            macd_hist > 0 and  # MACD histogram positive
            current_price > bollinger_upper * 0.998  # Near or above Bollinger upper band
        )
        
        if strong_momentum and rsi < 80:  # Not extremely overbought
            confidence = min(0.85, (rsi_fast / 100 + (macd_hist / abs(macd_hist) if macd_hist != 0 else 0)) / 2)
            
            return MarketOpportunity(
                pair=pair,
                opportunity_type=OpportunityType.MOMENTUM_BREAKOUT,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=current_price - (atr * 1.5),  # Tight stop
                take_profit=current_price + (atr * 3.0),  # Aggressive target
                position_size=0.12,  # Large position for high confidence
                expected_return=(atr * 3.0) / current_price * 100,
                risk_score=0.3,  # Medium risk for high reward
                timestamp=pd.Timestamp.now()
            )
        return None
    
    def _detect_reversal_opportunity(self, df: pd.DataFrame, indicators: Dict,
                                   current_price: float, atr: float, pair: str) -> Optional[MarketOpportunity]:
        """Detect reversal bounce opportunities at support levels"""
        rsi = indicators['rsi'][-1]
        williams_r = indicators['williams_r'][-1]
        bollinger_lower = indicators['bollinger_lower'][-1]
        pivot_lows = indicators['pivot_low']
        
        # Find recent support level
        recent_support = None
        for i in range(len(pivot_lows)-1, max(0, len(pivot_lows)-20), -1):
            if pivot_lows[i] > 0 and pivot_lows[i] <= current_price * 1.02:
                recent_support = pivot_lows[i]
                break
        
        # Oversold reversal conditions (aggressive)
        oversold_reversal = (
            rsi < 30 and  # Oversold
            williams_r < -75 and  # Oversold on Williams %R
            current_price <= bollinger_lower * 1.005  # Near Bollinger lower band
        )
        
        if oversold_reversal and recent_support:
            confidence = min(0.80, (30 - rsi) / 30 + abs(williams_r + 75) / 25) / 2
            
            return MarketOpportunity(
                pair=pair,
                opportunity_type=OpportunityType.REVERSAL_BOUNCE,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=recent_support * 0.98,  # Below support
                take_profit=current_price + (atr * 2.5),  # Aggressive target
                position_size=0.10,  # Good size for reversal play
                expected_return=(atr * 2.5) / current_price * 100,
                risk_score=0.4,  # Higher risk, but good reward potential
                timestamp=pd.Timestamp.now()
            )
        return None
    
    def _detect_volatility_opportunity(self, df: pd.DataFrame, indicators: Dict, 
                                     market_structure: Dict, current_price: float, 
                                     atr: float, pair: str) -> Optional[MarketOpportunity]:
        """Detect opportunities during volatility expansion"""
        if not market_structure.get("volatility_expansion", False):
            return None
            
        current_volatility = market_structure["volatility"]
        trend_strength = market_structure["trend_strength"]
        
        # High volatility with strong trend = opportunity
        if current_volatility > 3.0 and trend_strength > 0.6:
            confidence = min(0.75, trend_strength * (current_volatility / 5.0))
            
            # Direction based on trend
            if market_structure["trend"] in [MarketTrend.STRONG_BULLISH, MarketTrend.BULLISH]:
                return MarketOpportunity(
                    pair=pair,
                    opportunity_type=OpportunityType.VOLATILITY_EXPANSION,
                    confidence=confidence,
                    entry_price=current_price,
                    stop_loss=current_price - (atr * 2.0),
                    take_profit=current_price + (atr * 4.0),  # Big target for volatility
                    position_size=0.08,  # Moderate size due to volatility
                    expected_return=(atr * 4.0) / current_price * 100,
                    risk_score=0.5,  # Higher risk due to volatility
                    timestamp=pd.Timestamp.now()
                )
        return None