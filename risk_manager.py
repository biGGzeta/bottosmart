"""
Advanced risk management for opportunistic trading
"""
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

from config import config

@dataclass
class RiskMetrics:
    """Risk assessment metrics"""
    current_drawdown: float
    max_drawdown: float
    daily_pnl: float
    portfolio_exposure: float
    var_95: float  # Value at Risk
    sharpe_ratio: float
    volatility: float

class RiskManager:
    """
    Comprehensive risk management system for opportunistic trading
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.daily_pnl_history = []
        self.portfolio_values = []
        self.max_portfolio_value = config.INITIAL_BALANCE
        
    def check_trading_conditions(self, session) -> bool:
        """
        Check if current conditions allow for trading
        """
        # Update portfolio tracking
        self.portfolio_values.append(session.portfolio_value)
        self.max_portfolio_value = max(self.max_portfolio_value, session.portfolio_value)
        
        # Calculate current risk metrics
        risk_metrics = self.calculate_risk_metrics(session)
        
        # Check risk limits
        risk_checks = [
            self._check_drawdown_limit(risk_metrics),
            self._check_daily_loss_limit(risk_metrics),
            self._check_portfolio_exposure(session),
            self._check_volatility_threshold(risk_metrics),
            self._check_consecutive_losses(session)
        ]
        
        # Log risk status
        if not all(risk_checks):
            failed_checks = []
            check_names = ['drawdown', 'daily_loss', 'exposure', 'volatility', 'consecutive_losses']
            for i, check in enumerate(risk_checks):
                if not check:
                    failed_checks.append(check_names[i])
                    
            self.logger.warning(f"⚠️ Risk management halt - Failed checks: {', '.join(failed_checks)}")
            return False
            
        return True
    
    def calculate_risk_metrics(self, session) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Current drawdown
        current_drawdown = (self.max_portfolio_value - session.portfolio_value) / self.max_portfolio_value
        
        # Daily PnL
        today = datetime.now().date()
        daily_pnl = self._calculate_daily_pnl(session, today)
        
        # Portfolio exposure (sum of all position sizes)
        total_exposure = 0.0
        # This would be calculated from active positions in the main bot
        
        # Value at Risk (95% confidence)
        var_95 = self._calculate_var(session) if len(self.portfolio_values) > 30 else 0.0
        
        # Sharpe ratio
        sharpe_ratio = self._calculate_sharpe_ratio() if len(self.portfolio_values) > 30 else 0.0
        
        # Volatility
        volatility = self._calculate_volatility() if len(self.portfolio_values) > 10 else 0.0
        
        return RiskMetrics(
            current_drawdown=current_drawdown,
            max_drawdown=session.max_drawdown,
            daily_pnl=daily_pnl,
            portfolio_exposure=total_exposure,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            volatility=volatility
        )
    
    def _check_drawdown_limit(self, metrics: RiskMetrics) -> bool:
        """Check if drawdown is within acceptable limits"""
        if metrics.current_drawdown > config.MAX_DRAWDOWN:
            self.logger.error(f"❌ Drawdown limit exceeded: {metrics.current_drawdown:.2%} > {config.MAX_DRAWDOWN:.2%}")
            return False
        return True
    
    def _check_daily_loss_limit(self, metrics: RiskMetrics) -> bool:
        """Check if daily loss is within acceptable limits"""
        if metrics.daily_pnl < -config.MAX_DAILY_LOSS:
            self.logger.error(f"❌ Daily loss limit exceeded: {metrics.daily_pnl:.2%} < {-config.MAX_DAILY_LOSS:.2%}")
            return False
        return True
    
    def _check_portfolio_exposure(self, session) -> bool:
        """Check total portfolio exposure"""
        # For opportunistic trading, we allow higher exposure but still need limits
        max_total_exposure = 0.8  # 80% maximum total exposure
        
        # This would be calculated from actual positions
        # For now, assume it's acceptable
        return True
    
    def _check_volatility_threshold(self, metrics: RiskMetrics) -> bool:
        """Check if portfolio volatility is acceptable"""
        max_volatility = 0.15  # 15% annualized volatility limit
        
        if metrics.volatility > max_volatility:
            self.logger.warning(f"⚠️ High portfolio volatility: {metrics.volatility:.2%}")
            # Don't halt trading for volatility, but log warning
            
        return True
    
    def _check_consecutive_losses(self, session) -> bool:
        """Check for consecutive losses pattern"""
        # If we have more than 5 consecutive losses, reduce trading
        if session.losing_trades > 5 and session.winning_trades == 0:
            self.logger.warning("⚠️ Multiple consecutive losses detected")
            return False
            
        return True
    
    def _calculate_daily_pnl(self, session, date) -> float:
        """Calculate PnL for specific date"""
        # Simplified calculation - would need more sophisticated tracking
        return session.total_pnl / max(1, (datetime.now() - session.start_time).days + 1)
    
    def _calculate_var(self, session) -> float:
        """Calculate Value at Risk at 95% confidence level"""
        if len(self.portfolio_values) < 30:
            return 0.0
            
        # Calculate daily returns
        values = np.array(self.portfolio_values[-30:])  # Last 30 days
        returns = np.diff(values) / values[:-1]
        
        # 95% VaR
        return np.percentile(returns, 5)
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio"""
        if len(self.portfolio_values) < 30:
            return 0.0
            
        values = np.array(self.portfolio_values[-30:])
        returns = np.diff(values) / values[:-1]
        
        if np.std(returns) == 0:
            return 0.0
            
        # Assuming risk-free rate of 2% annually (0.0055% daily)
        risk_free_rate = 0.000055
        excess_returns = returns - risk_free_rate
        
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_volatility(self) -> float:
        """Calculate portfolio volatility"""
        if len(self.portfolio_values) < 10:
            return 0.0
            
        values = np.array(self.portfolio_values[-30:])  # Last 30 observations
        returns = np.diff(values) / values[:-1]
        
        return np.std(returns) * np.sqrt(252)  # Annualized volatility
    
    def calculate_position_risk(self, entry_price: float, stop_loss: float, 
                              position_size: float, current_price: float = None) -> Dict:
        """Calculate risk metrics for a specific position"""
        if current_price is None:
            current_price = entry_price
            
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        # Total position risk
        total_risk = risk_per_share * position_size
        
        # Risk as percentage of portfolio
        risk_percentage = total_risk / config.INITIAL_BALANCE
        
        # Current unrealized PnL
        unrealized_pnl = (current_price - entry_price) * position_size
        
        return {
            'risk_per_share': risk_per_share,
            'total_risk': total_risk,
            'risk_percentage': risk_percentage,
            'unrealized_pnl': unrealized_pnl,
            'risk_reward_ratio': abs((entry_price - stop_loss) / (current_price - entry_price)) if current_price != entry_price else 0
        }
    
    def suggest_position_adjustments(self, session) -> List[str]:
        """Suggest position adjustments based on risk analysis"""
        suggestions = []
        
        risk_metrics = self.calculate_risk_metrics(session)
        
        # High drawdown
        if risk_metrics.current_drawdown > config.MAX_DRAWDOWN * 0.7:
            suggestions.append("Consider reducing position sizes due to high drawdown")
            
        # High volatility
        if risk_metrics.volatility > 0.12:
            suggestions.append("High portfolio volatility detected - consider tighter stops")
            
        # Low Sharpe ratio
        if risk_metrics.sharpe_ratio < 1.0 and len(self.portfolio_values) > 30:
            suggestions.append("Low risk-adjusted returns - review strategy effectiveness")
            
        # Consecutive losses
        if session.losing_trades > 3 and session.winning_trades == 0:
            suggestions.append("Consider reducing trading frequency after consecutive losses")
            
        return suggestions
    
    def emergency_stop_conditions(self, session) -> bool:
        """Check for conditions requiring emergency stop"""
        risk_metrics = self.calculate_risk_metrics(session)
        
        emergency_conditions = [
            risk_metrics.current_drawdown > config.MAX_DRAWDOWN * 1.2,  # 20% over limit
            risk_metrics.daily_pnl < -config.MAX_DAILY_LOSS * 1.5,     # 50% over daily limit
            session.losing_trades > 10 and session.winning_trades == 0, # Too many consecutive losses
        ]
        
        if any(emergency_conditions):
            self.logger.critical("🚨 EMERGENCY STOP CONDITIONS MET - HALTING ALL TRADING")
            return True
            
        return False
    
    def get_risk_summary(self, session) -> Dict:
        """Get comprehensive risk summary"""
        risk_metrics = self.calculate_risk_metrics(session)
        
        return {
            'current_drawdown': risk_metrics.current_drawdown,
            'daily_pnl': risk_metrics.daily_pnl,
            'volatility': risk_metrics.volatility,
            'sharpe_ratio': risk_metrics.sharpe_ratio,
            'var_95': risk_metrics.var_95,
            'suggestions': self.suggest_position_adjustments(session),
            'emergency_stop_required': self.emergency_stop_conditions(session)
        }