"""
Core trading bot engine with adaptive and opportunistic strategies
"""
import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from enum import Enum

from config import config
from market_analyzer import MarketAnalyzer, MarketOpportunity, OpportunityType, MarketTrend
from risk_manager import RiskManager
from ml_predictor import MLPredictor
from exchange_connector import ExchangeConnector

class BotState(Enum):
    IDLE = "idle"
    SCANNING = "scanning"
    TRADING = "trading"
    RISK_MANAGEMENT = "risk_management"
    EMERGENCY_STOP = "emergency_stop"

@dataclass
class Position:
    """Trading position"""
    pair: str
    side: str  # 'buy' or 'sell'
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    entry_time: datetime
    opportunity_type: OpportunityType
    current_pnl: float = 0.0
    max_pnl: float = 0.0
    min_pnl: float = 0.0

@dataclass
class TradingSession:
    """Trading session statistics"""
    start_time: datetime = field(default_factory=datetime.now)
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    portfolio_value: float = config.INITIAL_BALANCE
    opportunities_detected: int = 0
    opportunities_taken: int = 0

class BottoSmart:
    """
    Intelligent and adaptive trading bot with opportunistic strategies
    """
    
    def __init__(self):
        # Core components
        self.market_analyzer = MarketAnalyzer()
        self.risk_manager = RiskManager()
        self.ml_predictor = MLPredictor()
        self.exchange = ExchangeConnector()
        
        # Bot state
        self.state = BotState.IDLE
        self.session = TradingSession()
        self.positions: Dict[str, Position] = {}
        self.market_data: Dict[str, pd.DataFrame] = {}
        
        # Adaptive parameters
        self.performance_tracker = {
            'recent_wins': [],
            'recent_losses': [],
            'strategy_performance': {},
            'market_conditions': []
        }
        
        # Setup logging
        self.setup_logging()
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bottosmart.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def run(self):
        """Main bot execution loop"""
        self.logger.info("🚀 BottoSmart starting - Opportunistic Trading Mode")
        self.logger.info(f"Trading pairs: {config.TRADING_PAIRS}")
        self.logger.info(f"Max position size: {config.MAX_POSITION_SIZE * 100}%")
        
        try:
            # Initialize exchange connection
            await self.exchange.connect()
            
            # Start main trading loop
            while True:
                await self.trading_cycle()
                await asyncio.sleep(5)  # Quick cycle for opportunistic trading
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Bot stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Critical error: {e}")
            self.state = BotState.EMERGENCY_STOP
        finally:
            await self.cleanup()
            
    async def trading_cycle(self):
        """Execute one complete trading cycle"""
        try:
            # Update market data
            await self.update_market_data()
            
            # Check current positions
            await self.manage_positions()
            
            # Risk management check
            if not self.risk_manager.check_trading_conditions(self.session):
                self.state = BotState.RISK_MANAGEMENT
                return
                
            # Scan for opportunities if we have capacity
            if len(self.positions) < 5:  # Max 5 concurrent positions
                self.state = BotState.SCANNING
                opportunities = await self.scan_opportunities()
                
                # Execute best opportunities
                if opportunities:
                    self.state = BotState.TRADING
                    await self.execute_opportunities(opportunities)
            
            # Adapt strategies based on performance
            self.adapt_strategies()
            
            # Log session stats periodically
            if self.session.total_trades > 0 and self.session.total_trades % 10 == 0:
                self.log_session_stats()
                
        except Exception as e:
            self.logger.error(f"Error in trading cycle: {e}")
            
    async def update_market_data(self):
        """Update market data for all trading pairs"""
        for pair in config.TRADING_PAIRS:
            try:
                # Get recent OHLCV data
                df = await self.exchange.get_ohlcv(pair, timeframe='5m', limit=200)
                if df is not None and len(df) > 0:
                    self.market_data[pair] = df
            except Exception as e:
                self.logger.error(f"Failed to update data for {pair}: {e}")
                
    async def scan_opportunities(self) -> List[MarketOpportunity]:
        """Scan all pairs for trading opportunities"""
        all_opportunities = []
        
        for pair in config.TRADING_PAIRS:
            if pair in self.market_data and len(self.market_data[pair]) > 50:
                try:
                    # Analyze market
                    analysis = self.market_analyzer.analyze_market_data(
                        self.market_data[pair], pair
                    )
                    
                    opportunities = analysis.get('opportunities', [])
                    self.session.opportunities_detected += len(opportunities)
                    
                    # Add ML prediction confidence
                    for opp in opportunities:
                        ml_confidence = await self.ml_predictor.predict_price_movement(
                            self.market_data[pair], opp.opportunity_type
                        )
                        
                        # Combine technical and ML confidence
                        opp.confidence = (opp.confidence * 0.6 + ml_confidence * 0.4)
                        
                        # Only consider high-confidence opportunities (opportunistic approach)
                        if opp.confidence >= config.PREDICTION_CONFIDENCE_THRESHOLD:
                            all_opportunities.append(opp)
                            
                except Exception as e:
                    self.logger.error(f"Error analyzing {pair}: {e}")
                    
        # Sort by confidence and expected return
        all_opportunities.sort(
            key=lambda x: x.confidence * x.expected_return, 
            reverse=True
        )
        
        return all_opportunities[:3]  # Top 3 opportunities
    
    async def execute_opportunities(self, opportunities: List[MarketOpportunity]):
        """Execute the best trading opportunities"""
        for opp in opportunities:
            try:
                # Skip if already have position in this pair
                if opp.pair in self.positions:
                    continue
                    
                # Calculate position size dynamically
                position_size = self.calculate_dynamic_position_size(opp)
                
                if position_size < config.MIN_POSITION_SIZE:
                    self.logger.info(f"Position size too small for {opp.pair}: {position_size}")
                    continue
                
                # Execute trade
                trade_result = await self.execute_trade(opp, position_size)
                
                if trade_result:
                    self.session.opportunities_taken += 1
                    self.logger.info(
                        f"✅ Opened {opp.opportunity_type.value} position in {opp.pair} "
                        f"(Confidence: {opp.confidence:.2f}, Size: {position_size:.2f})"
                    )
                    
            except Exception as e:
                self.logger.error(f"Error executing opportunity {opp.pair}: {e}")
                
    def calculate_dynamic_position_size(self, opp: MarketOpportunity) -> float:
        """Calculate position size based on confidence, volatility, and performance"""
        base_size = config.MAX_POSITION_SIZE
        
        # Adjust based on confidence (opportunistic approach)
        confidence_multiplier = min(1.5, opp.confidence / 0.5)  # Up to 150% for high confidence
        
        # Adjust based on recent performance
        recent_performance = self.get_recent_performance()
        if recent_performance > 0.02:  # If doing well, be more aggressive
            performance_multiplier = 1.2
        elif recent_performance < -0.02:  # If doing poorly, be more conservative
            performance_multiplier = 0.7
        else:
            performance_multiplier = 1.0
            
        # Adjust based on opportunity type
        type_multiplier = {
            OpportunityType.MOMENTUM_BREAKOUT: 1.3,  # Most aggressive
            OpportunityType.VOLATILITY_EXPANSION: 1.1,
            OpportunityType.TREND_CONTINUATION: 1.0,
            OpportunityType.REVERSAL_BOUNCE: 0.9    # Slightly more conservative
        }.get(opp.opportunity_type, 1.0)
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * performance_multiplier * type_multiplier
        
        # Ensure within bounds
        return min(max(position_size, config.MIN_POSITION_SIZE), config.MAX_POSITION_SIZE)
    
    async def execute_trade(self, opp: MarketOpportunity, position_size: float) -> bool:
        """Execute a single trade"""
        try:
            # Calculate quantity
            portfolio_value = self.session.portfolio_value
            trade_amount = portfolio_value * position_size
            quantity = trade_amount / opp.entry_price
            
            # Place order
            order_result = await self.exchange.create_order(
                symbol=opp.pair,
                type='market',
                side='buy',  # Assuming long positions for now
                amount=quantity,
                params={'reduceOnly': False}
            )
            
            if order_result and order_result.get('status') == 'filled':
                # Create position record
                position = Position(
                    pair=opp.pair,
                    side='buy',
                    entry_price=order_result.get('average', opp.entry_price),
                    quantity=quantity,
                    stop_loss=opp.stop_loss,
                    take_profit=opp.take_profit,
                    entry_time=datetime.now(),
                    opportunity_type=opp.opportunity_type
                )
                
                self.positions[opp.pair] = position
                self.session.total_trades += 1
                
                # Place stop loss and take profit orders
                await self.place_exit_orders(position)
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to execute trade for {opp.pair}: {e}")
            
        return False
    
    async def place_exit_orders(self, position: Position):
        """Place stop loss and take profit orders"""
        try:
            # Stop loss order
            await self.exchange.create_order(
                symbol=position.pair,
                type='stop_market',
                side='sell',
                amount=position.quantity,
                params={
                    'stopPrice': position.stop_loss,
                    'reduceOnly': True
                }
            )
            
            # Take profit order
            await self.exchange.create_order(
                symbol=position.pair,
                type='limit',
                side='sell',
                amount=position.quantity,
                price=position.take_profit,
                params={'reduceOnly': True}
            )
            
        except Exception as e:
            self.logger.error(f"Failed to place exit orders for {position.pair}: {e}")
            
    async def manage_positions(self):
        """Manage open positions"""
        for pair, position in list(self.positions.items()):
            try:
                # Get current price
                current_price = await self.exchange.get_current_price(pair)
                if current_price is None:
                    continue
                    
                # Update PnL
                pnl = (current_price - position.entry_price) / position.entry_price
                position.current_pnl = pnl * 100  # Convert to percentage
                position.max_pnl = max(position.max_pnl, position.current_pnl)
                position.min_pnl = min(position.min_pnl, position.current_pnl)
                
                # Check for manual exit conditions (adaptive)
                should_exit, reason = self.should_exit_position(position, current_price)
                
                if should_exit:
                    await self.close_position(position, reason)
                    
            except Exception as e:
                self.logger.error(f"Error managing position {pair}: {e}")
                
    def should_exit_position(self, position: Position, current_price: float) -> Tuple[bool, str]:
        """Determine if position should be closed manually"""
        # Time-based exit (for momentum trades)
        if position.opportunity_type == OpportunityType.MOMENTUM_BREAKOUT:
            time_held = (datetime.now() - position.entry_time).seconds / 3600  # hours
            if time_held > 4:  # Close momentum trades after 4 hours
                return True, "time_limit_momentum"
                
        # Trailing stop for profitable trades
        if position.current_pnl > 3.0 and position.current_pnl < position.max_pnl * 0.7:
            return True, "trailing_stop"
            
        # Emergency exit on severe adverse movement
        if position.current_pnl < -4.0:  # Wider than normal stop for opportunities
            return True, "emergency_stop"
            
        return False, ""
    
    async def close_position(self, position: Position, reason: str):
        """Close a position"""
        try:
            # Cancel existing orders
            await self.exchange.cancel_all_orders(position.pair)
            
            # Market sell
            order_result = await self.exchange.create_order(
                symbol=position.pair,
                type='market',
                side='sell',
                amount=position.quantity,
                params={'reduceOnly': True}
            )
            
            if order_result and order_result.get('status') == 'filled':
                # Update session stats
                if position.current_pnl > 0:
                    self.session.winning_trades += 1
                else:
                    self.session.losing_trades += 1
                    
                self.session.total_pnl += position.current_pnl
                
                # Track performance
                self.performance_tracker['recent_wins' if position.current_pnl > 0 else 'recent_losses'].append({
                    'pnl': position.current_pnl,
                    'type': position.opportunity_type,
                    'reason': reason,
                    'timestamp': datetime.now()
                })
                
                self.logger.info(
                    f"🔄 Closed {position.pair} - PnL: {position.current_pnl:.2f}% "
                    f"({reason}) - Session PnL: {self.session.total_pnl:.2f}%"
                )
                
                # Remove from positions
                del self.positions[position.pair]
                
        except Exception as e:
            self.logger.error(f"Failed to close position {position.pair}: {e}")
            
    def adapt_strategies(self):
        """Adapt trading strategies based on performance"""
        # Keep only recent performance data (last 20 trades)
        for key in ['recent_wins', 'recent_losses']:
            self.performance_tracker[key] = self.performance_tracker[key][-20:]
            
        # Adjust aggressiveness based on recent performance
        recent_performance = self.get_recent_performance()
        
        if recent_performance > 0.05:  # Doing very well
            # Increase position sizes and lower confidence threshold
            config.MAX_POSITION_SIZE = min(0.20, config.MAX_POSITION_SIZE * 1.05)
            config.PREDICTION_CONFIDENCE_THRESHOLD = max(0.55, config.PREDICTION_CONFIDENCE_THRESHOLD * 0.98)
        elif recent_performance < -0.05:  # Doing poorly
            # Decrease position sizes and raise confidence threshold
            config.MAX_POSITION_SIZE = max(0.08, config.MAX_POSITION_SIZE * 0.95)
            config.PREDICTION_CONFIDENCE_THRESHOLD = min(0.75, config.PREDICTION_CONFIDENCE_THRESHOLD * 1.02)
            
    def get_recent_performance(self) -> float:
        """Get recent performance metric"""
        wins = self.performance_tracker['recent_wins']
        losses = self.performance_tracker['recent_losses']
        
        if not wins and not losses:
            return 0.0
            
        all_trades = wins + losses
        if len(all_trades) < 5:  # Need minimum trades for meaningful metric
            return 0.0
            
        # Recent 10 trades performance
        recent_trades = sorted(all_trades, key=lambda x: x['timestamp'])[-10:]
        return sum(trade['pnl'] for trade in recent_trades) / len(recent_trades)
    
    def log_session_stats(self):
        """Log comprehensive session statistics"""
        win_rate = (self.session.winning_trades / self.session.total_trades * 100) if self.session.total_trades > 0 else 0
        opportunity_rate = (self.session.opportunities_taken / self.session.opportunities_detected * 100) if self.session.opportunities_detected > 0 else 0
        
        self.logger.info(
            f"📊 Session Stats - "
            f"Trades: {self.session.total_trades} "
            f"Win Rate: {win_rate:.1f}% "
            f"PnL: {self.session.total_pnl:.2f}% "
            f"Opportunities: {self.session.opportunities_detected} "
            f"Taken: {opportunity_rate:.1f}% "
            f"Active Positions: {len(self.positions)}"
        )
        
    async def cleanup(self):
        """Cleanup resources"""
        self.logger.info("🧹 Cleaning up...")
        
        # Close all positions
        for position in list(self.positions.values()):
            await self.close_position(position, "shutdown")
            
        # Disconnect from exchange
        await self.exchange.disconnect()
        
        # Final session report
        self.log_session_stats()
        self.logger.info("✅ Cleanup complete")

# Main execution
async def main():
    bot = BottoSmart()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())