#!/usr/bin/env python3
"""
BottoSmart Demo - Showcase the bot's intelligent features
This demo runs without external dependencies to show the bot's logic
"""

import random
import time
from datetime import datetime, timedelta
from typing import Dict, List

class MockMarketData:
    """Generate realistic market data for demo"""
    
    def __init__(self):
        self.price = 45000  # Starting BTC price
        self.trend = 1
        self.volatility = 0.02
        
    def generate_candle(self):
        """Generate a realistic OHLC candle"""
        # Random walk with trend
        change = random.gauss(0, self.volatility) + (self.trend * 0.001)
        new_price = self.price * (1 + change)
        
        # Create OHLC
        high = new_price * (1 + random.uniform(0, 0.01))
        low = new_price * (1 - random.uniform(0, 0.01))
        close = new_price
        open_price = self.price
        
        self.price = close
        
        # Occasionally reverse trend
        if random.random() < 0.05:
            self.trend *= -1
            
        return {
            'timestamp': datetime.now(),
            'open': open_price,
            'high': high,
            'low': low,
            'close': close,
            'volume': random.uniform(100, 1000)
        }

class BottoSmartDemo:
    """Demo version of BottoSmart showing key features"""
    
    def __init__(self):
        self.balance = 10000.0
        self.positions = []
        self.trades = []
        self.market_data = MockMarketData()
        self.opportunities_detected = 0
        self.opportunities_taken = 0
        
        # Demo configuration
        self.max_position_size = 0.15  # 15% aggressive sizing
        self.confidence_threshold = 0.65
        self.stop_loss_pct = 2.5
        self.take_profit_pct = 6.0
        
    def analyze_market(self, candle: Dict) -> Dict:
        """Simulate advanced market analysis"""
        print(f"  🔍 Analyzing BTC/USDT @ ${candle['close']:.2f}")
        
        # Simulate technical indicators
        rsi = random.uniform(20, 80)
        macd = random.uniform(-100, 100)
        bb_position = random.uniform(0, 1)
        volume_ratio = random.uniform(0.5, 2.0)
        
        # Simulate opportunity detection
        opportunities = []
        
        # Momentum breakout simulation
        if rsi > 60 and macd > 0 and bb_position > 0.8:
            confidence = min(0.9, (rsi / 100 + macd / 100 + bb_position) / 3)
            opportunities.append({
                'type': 'momentum_breakout',
                'confidence': confidence,
                'entry_price': candle['close'],
                'stop_loss': candle['close'] * (1 - self.stop_loss_pct / 100),
                'take_profit': candle['close'] * (1 + self.take_profit_pct / 100)
            })
            print(f"  📈 Momentum breakout detected (Confidence: {confidence:.2f})")
            
        # Reversal bounce simulation
        elif rsi < 35 and random.random() > 0.7:
            confidence = min(0.85, (35 - rsi) / 35 + random.uniform(0.2, 0.4))
            opportunities.append({
                'type': 'reversal_bounce',
                'confidence': confidence,
                'entry_price': candle['close'],
                'stop_loss': candle['close'] * (1 - self.stop_loss_pct / 100),
                'take_profit': candle['close'] * (1 + self.take_profit_pct / 100)
            })
            print(f"  🔄 Reversal bounce detected (Confidence: {confidence:.2f})")
            
        return {
            'rsi': rsi,
            'macd': macd,
            'bb_position': bb_position,
            'volume_ratio': volume_ratio,
            'opportunities': opportunities
        }
    
    def ml_prediction(self, opportunity: Dict) -> float:
        """Simulate ML prediction"""
        base_confidence = opportunity['confidence']
        
        # Simulate ML model prediction
        ml_confidence = random.uniform(0.4, 0.8)
        
        # Combine technical and ML
        combined = base_confidence * 0.6 + ml_confidence * 0.4
        
        print(f"  🧠 ML Prediction: {ml_confidence:.2f} | Combined: {combined:.2f}")
        return combined
    
    def calculate_position_size(self, opportunity: Dict) -> float:
        """Calculate dynamic position size"""
        base_size = self.max_position_size
        
        # Adjust based on confidence (opportunistic approach)
        confidence_multiplier = min(1.5, opportunity['confidence'] / 0.5)
        
        # Simulate recent performance adjustment
        recent_performance = random.uniform(-0.05, 0.05)
        if recent_performance > 0.02:
            performance_multiplier = 1.2  # More aggressive if doing well
        elif recent_performance < -0.02:
            performance_multiplier = 0.7  # More conservative if doing poorly
        else:
            performance_multiplier = 1.0
            
        position_size = base_size * confidence_multiplier * performance_multiplier
        return min(max(position_size, 0.02), self.max_position_size)
    
    def execute_trade(self, opportunity: Dict, position_size: float, candle: Dict):
        """Execute a trade"""
        trade_amount = self.balance * position_size
        quantity = trade_amount / opportunity['entry_price']
        
        position = {
            'type': opportunity['type'],
            'entry_price': opportunity['entry_price'],
            'quantity': quantity,
            'trade_amount': trade_amount,
            'stop_loss': opportunity['stop_loss'],
            'take_profit': opportunity['take_profit'],
            'entry_time': candle['timestamp'],
            'confidence': opportunity['confidence']
        }
        
        self.positions.append(position)
        self.balance -= trade_amount
        self.opportunities_taken += 1
        
        print(f"  ✅ TRADE EXECUTED: {opportunity['type']}")
        print(f"     Size: {position_size:.1%} (${trade_amount:.2f})")
        print(f"     Entry: ${opportunity['entry_price']:.2f}")
        print(f"     Stop: ${opportunity['stop_loss']:.2f}")
        print(f"     Target: ${opportunity['take_profit']:.2f}")
    
    def manage_positions(self, candle: Dict):
        """Manage open positions"""
        for position in self.positions[:]:  # Copy to modify during iteration
            current_price = candle['close']
            
            # Check exit conditions
            if current_price <= position['stop_loss']:
                self.close_position(position, current_price, 'stop_loss')
            elif current_price >= position['take_profit']:
                self.close_position(position, current_price, 'take_profit')
            elif random.random() < 0.05:  # Occasional manual exit
                self.close_position(position, current_price, 'manual_exit')
    
    def close_position(self, position: Dict, exit_price: float, reason: str):
        """Close a position and calculate PnL"""
        pnl = (exit_price - position['entry_price']) / position['entry_price']
        trade_return = position['trade_amount'] * (1 + pnl)
        
        self.balance += trade_return
        self.positions.remove(position)
        
        trade_record = {
            'type': position['type'],
            'entry_price': position['entry_price'],
            'exit_price': exit_price,
            'pnl_pct': pnl * 100,
            'pnl_dollar': trade_return - position['trade_amount'],
            'reason': reason,
            'confidence': position['confidence']
        }
        
        self.trades.append(trade_record)
        
        print(f"  📤 POSITION CLOSED: {reason}")
        print(f"     PnL: {pnl*100:+.2f}% (${trade_record['pnl_dollar']:+.2f})")
        print(f"     Balance: ${self.balance:.2f}")
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        if not self.trades:
            return {'total_return': 0, 'win_rate': 0, 'total_trades': 0}
            
        total_return = (self.balance - 10000) / 10000 * 100
        winning_trades = len([t for t in self.trades if t['pnl_pct'] > 0])
        win_rate = winning_trades / len(self.trades) * 100
        
        return {
            'total_return': total_return,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'opportunities_detected': self.opportunities_detected,
            'opportunities_taken': self.opportunities_taken,
            'active_positions': len(self.positions)
        }
    
    def run_demo(self, cycles: int = 20):
        """Run the demo simulation"""
        print("🚀 BottoSmart v2.0 - Opportunistic Trading Demo")
        print("=" * 60)
        print(f"Initial Balance: ${self.balance:,.2f}")
        print(f"Max Position Size: {self.max_position_size:.1%} (Aggressive)")
        print(f"Confidence Threshold: {self.confidence_threshold:.1%}")
        print("=" * 60)
        
        for cycle in range(cycles):
            print(f"\n📊 Cycle {cycle + 1}/{cycles}")
            
            # Generate market data
            candle = self.market_data.generate_candle()
            
            # Manage existing positions
            if self.positions:
                self.manage_positions(candle)
            
            # Look for new opportunities
            if len(self.positions) < 3:  # Max 3 positions
                analysis = self.analyze_market(candle)
                opportunities = analysis['opportunities']
                self.opportunities_detected += len(opportunities)
                
                for opp in opportunities:
                    # Add ML prediction
                    ml_confidence = self.ml_prediction(opp)
                    opp['confidence'] = ml_confidence
                    
                    # Check if opportunity meets threshold
                    if opp['confidence'] >= self.confidence_threshold:
                        position_size = self.calculate_position_size(opp)
                        self.execute_trade(opp, position_size, candle)
                        break  # One trade per cycle
            
            # Brief pause for readability
            time.sleep(0.5)
        
        # Final statistics
        stats = self.get_stats()
        print("\n" + "=" * 60)
        print("📈 DEMO RESULTS")
        print("=" * 60)
        print(f"🏆 Total Return: {stats['total_return']:+.2f}%")
        print(f"💰 Final Balance: ${self.balance:,.2f}")
        print(f"📊 Total Trades: {stats['total_trades']}")
        print(f"🎯 Win Rate: {stats['win_rate']:.1f}%")
        print(f"🔍 Opportunities Detected: {stats['opportunities_detected']}")
        print(f"✅ Opportunities Taken: {stats['opportunities_taken']}")
        print(f"📍 Active Positions: {stats['active_positions']}")
        print("=" * 60)
        
        if stats['total_return'] > 0:
            print("🎉 Profitable demo run! The opportunistic strategy shows potential.")
        else:
            print("📊 Demo shows the adaptive risk management in action.")
        
        print("\nℹ️  This was a demonstration with simulated data.")
        print("📚 Real trading requires proper setup with exchange APIs.")

if __name__ == "__main__":
    demo = BottoSmartDemo()
    
    print("Welcome to BottoSmart Demo!")
    print("This showcases the bot's opportunistic and intelligent features.")
    
    input("\nPress Enter to start the demo...")
    
    demo.run_demo(cycles=15)
    
    print("\n🔧 To run the real bot:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Configure .env with your API keys")
    print("3. Run: python main.py --sandbox")
    print("\n🚨 Always test in sandbox mode first!")