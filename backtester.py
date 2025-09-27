"""
Simple backtesting module for BottoSmart strategies
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from datetime import datetime, timedelta

from market_analyzer import MarketAnalyzer, MarketOpportunity
from ml_predictor import MLPredictor
from config import config

class SimpleBacktester:
    """
    Simple backtesting engine for BottoSmart strategies
    """
    
    def __init__(self, initial_balance: float = 10000):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions = []
        self.trades = []
        self.analyzer = MarketAnalyzer()
        self.ml_predictor = MLPredictor()
        self.logger = logging.getLogger(__name__)
        
    def backtest(self, df: pd.DataFrame, pair: str) -> Dict:
        """
        Run backtest on historical data
        """
        self.logger.info(f"🔍 Running backtest for {pair}")
        
        if len(df) < 100:
            return {"error": "Insufficient data"}
            
        # Reset state
        self.current_balance = self.initial_balance
        self.positions = []
        self.trades = []
        
        # Walk through data
        for i in range(100, len(df)):  # Start after sufficient lookback
            current_data = df.iloc[:i+1]
            current_price = current_data['close'].iloc[-1]
            timestamp = current_data.index[-1]
            
            # Check for exit signals on existing positions
            self._check_exits(current_price, timestamp)
            
            # Look for new opportunities if no position
            if not self.positions:
                analysis = self.analyzer.analyze_market_data(current_data, pair)
                opportunities = analysis.get('opportunities', [])
                
                # Take best opportunity
                if opportunities:
                    best_opp = max(opportunities, key=lambda x: x.confidence)
                    if best_opp.confidence >= 0.6:  # Lower threshold for backtesting
                        self._enter_position(best_opp, current_price, timestamp)
        
        # Close any remaining positions
        if self.positions:
            final_price = df['close'].iloc[-1]
            final_timestamp = df.index[-1]
            self._close_position(final_price, final_timestamp, "backtest_end")
            
        # Calculate results
        return self._calculate_results()
    
    def _enter_position(self, opportunity: MarketOpportunity, price: float, timestamp):
        """Enter a position based on opportunity"""
        position_size = min(config.MAX_POSITION_SIZE, 0.1)  # Conservative for backtest
        trade_amount = self.current_balance * position_size
        quantity = trade_amount / price
        
        position = {
            'pair': opportunity.pair,
            'entry_price': price,
            'quantity': quantity,
            'entry_time': timestamp,
            'stop_loss': opportunity.stop_loss,
            'take_profit': opportunity.take_profit,
            'opportunity_type': opportunity.opportunity_type,
            'trade_amount': trade_amount
        }
        
        self.positions.append(position)
        self.current_balance -= trade_amount
        
        self.logger.debug(f"📈 Entered position: {opportunity.pair} @ ${price:.4f}")
    
    def _check_exits(self, current_price: float, timestamp):
        """Check if any positions should be closed"""
        for position in self.positions[:]:  # Copy list to modify during iteration
            # Check stop loss
            if current_price <= position['stop_loss']:
                self._close_position(current_price, timestamp, "stop_loss")
                
            # Check take profit
            elif current_price >= position['take_profit']:
                self._close_position(current_price, timestamp, "take_profit")
                
            # Check time-based exit (optional)
            elif (timestamp - position['entry_time']).total_seconds() > 14400:  # 4 hours
                self._close_position(current_price, timestamp, "time_exit")
    
    def _close_position(self, exit_price: float, timestamp, reason: str):
        """Close the current position"""
        if not self.positions:
            return
            
        position = self.positions.pop(0)  # Close first position
        
        # Calculate PnL
        pnl = (exit_price - position['entry_price']) / position['entry_price']
        trade_return = position['trade_amount'] * (1 + pnl)
        self.current_balance += trade_return
        
        # Record trade
        trade = {
            'pair': position['pair'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'entry_time': position['entry_time'],
            'exit_time': timestamp,
            'quantity': position['quantity'],
            'pnl_percentage': pnl * 100,
            'pnl_dollar': trade_return - position['trade_amount'],
            'reason': reason,
            'opportunity_type': position['opportunity_type']
        }
        
        self.trades.append(trade)
        
        self.logger.debug(f"📉 Closed position: {position['pair']} @ ${exit_price:.4f} ({reason}) - PnL: {pnl*100:.2f}%")
    
    def _calculate_results(self) -> Dict:
        """Calculate comprehensive backtest results"""
        if not self.trades:
            return {
                "total_return": 0,
                "trades": 0,
                "win_rate": 0,
                "max_drawdown": 0,
                "sharpe_ratio": 0
            }
        
        # Basic metrics
        total_return = (self.current_balance - self.initial_balance) / self.initial_balance * 100
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['pnl_percentage'] > 0])
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
        
        # Average returns
        avg_win = np.mean([t['pnl_percentage'] for t in self.trades if t['pnl_percentage'] > 0])
        avg_loss = np.mean([t['pnl_percentage'] for t in self.trades if t['pnl_percentage'] <= 0])
        
        # Calculate drawdown
        balance_curve = [self.initial_balance]
        running_balance = self.initial_balance
        
        for trade in self.trades:
            running_balance += trade['pnl_dollar']
            balance_curve.append(running_balance)
        
        peak = balance_curve[0]
        max_drawdown = 0
        
        for balance in balance_curve:
            if balance > peak:
                peak = balance
            drawdown = (peak - balance) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)
        
        # Sharpe ratio (simplified)
        returns = [t['pnl_percentage'] for t in self.trades]
        if len(returns) > 1 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252/len(returns))
        else:
            sharpe_ratio = 0
        
        return {
            "total_return": round(total_return, 2),
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": total_trades - winning_trades,
            "win_rate": round(win_rate, 2),
            "avg_win": round(avg_win, 2) if not np.isnan(avg_win) else 0,
            "avg_loss": round(avg_loss, 2) if not np.isnan(avg_loss) else 0,
            "max_drawdown": round(max_drawdown, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "final_balance": round(self.current_balance, 2),
            "profit_factor": round(abs(avg_win / avg_loss), 2) if avg_loss < 0 else 0
        }
    
    def print_results(self, results: Dict):
        """Print formatted backtest results"""
        print(f"""
📊 Backtest Results
═══════════════════════════════════════
💰 Total Return: {results['total_return']}%
📈 Final Balance: ${results['final_balance']}
🎯 Total Trades: {results['total_trades']}
✅ Win Rate: {results['win_rate']}%
📊 Average Win: {results['avg_win']}%
📉 Average Loss: {results['avg_loss']}%
⬇️ Max Drawdown: {results['max_drawdown']}%
📈 Sharpe Ratio: {results['sharpe_ratio']}
⚖️ Profit Factor: {results['profit_factor']}
═══════════════════════════════════════
        """)

# Standalone backtesting script
async def run_backtest():
    """Run backtest on sample data"""
    from exchange_connector import ExchangeConnector
    
    backtester = SimpleBacktester(initial_balance=10000)
    exchange = ExchangeConnector()
    
    try:
        await exchange.connect()
        
        # Get historical data
        pairs_to_test = ['BTC/USDT', 'ETH/USDT']
        
        for pair in pairs_to_test:
            print(f"\n🔍 Testing {pair}...")
            df = await exchange.get_ohlcv(pair, timeframe='5m', limit=1000)
            
            if df is not None and len(df) > 200:
                results = backtester.backtest(df, pair)
                backtester.print_results(results)
            else:
                print(f"❌ Insufficient data for {pair}")
                
    except Exception as e:
        print(f"❌ Backtest error: {e}")
    finally:
        await exchange.disconnect()

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_backtest())