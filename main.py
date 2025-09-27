#!/usr/bin/env python3
"""
BottoSmart - Intelligent Opportunistic Trading Bot
Entry point for the application
"""

import asyncio
import argparse
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from bottosmart import BottoSmart
from config import config
from market_analyzer import MarketAnalyzer
from ml_predictor import MLPredictor
from exchange_connector import ExchangeConnector

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('bottosmart.log'),
            logging.StreamHandler()
        ]
    )

async def test_components():
    """Test all components before starting the bot"""
    print("🧪 Testing BottoSmart components...")
    
    # Test exchange connectivity
    exchange = ExchangeConnector()
    if await exchange.test_connectivity():
        print("✅ Exchange connectivity: OK")
    else:
        print("❌ Exchange connectivity: FAILED")
        return False
    
    # Test market analyzer
    try:
        analyzer = MarketAnalyzer()
        print("✅ Market analyzer: OK")
    except Exception as e:
        print(f"❌ Market analyzer: FAILED - {e}")
        return False
    
    # Test ML predictor
    try:
        predictor = MLPredictor()
        print("✅ ML predictor: OK")
    except Exception as e:
        print(f"❌ ML predictor: FAILED - {e}")
        return False
    
    await exchange.disconnect()
    print("✅ All components tested successfully!")
    return True

async def train_models():
    """Train ML models with historical data"""
    print("🎓 Training ML models...")
    
    exchange = ExchangeConnector()
    predictor = MLPredictor()
    
    try:
        await exchange.connect()
        
        # Get historical data for training
        for pair in config.TRADING_PAIRS[:3]:  # Train on first 3 pairs
            print(f"📊 Getting data for {pair}...")
            df = await exchange.get_ohlcv(pair, timeframe='5m', limit=1000)
            
            if df is not None and len(df) > 200:
                print(f"🎯 Training models for {pair}...")
                from market_analyzer import OpportunityType
                
                for opp_type in OpportunityType:
                    predictor.train_models(df, opp_type)
            else:
                print(f"⚠️ Insufficient data for {pair}")
        
        print("✅ Model training completed!")
        
    except Exception as e:
        print(f"❌ Model training failed: {e}")
    finally:
        await exchange.disconnect()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='BottoSmart - Opportunistic Trading Bot')
    parser.add_argument('--mode', choices=['run', 'test', 'train'], default='run',
                       help='Operation mode (default: run)')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       default='INFO', help='Logging level')
    parser.add_argument('--sandbox', action='store_true', 
                       help='Force sandbox mode')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Override sandbox if specified
    if args.sandbox:
        config.SANDBOX = True
    
    print(f"""
🚀 BottoSmart v2.0 - Intelligent Opportunistic Trading Bot
════════════════════════════════════════════════════════════
Exchange: {config.EXCHANGE}
Mode: {"SANDBOX" if config.SANDBOX else "LIVE"}
Max Position Size: {config.MAX_POSITION_SIZE * 100}%
Trading Pairs: {', '.join(config.TRADING_PAIRS)}
════════════════════════════════════════════════════════════
    """)
    
    if not config.SANDBOX:
        confirmation = input("⚠️  You are running in LIVE mode with real money. Continue? (yes/no): ")
        if confirmation.lower() != 'yes':
            print("🛑 Aborted by user")
            return
    
    # Run based on mode
    if args.mode == 'test':
        asyncio.run(test_components())
    elif args.mode == 'train':
        asyncio.run(train_models())
    else:  # run
        try:
            bot = BottoSmart()
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            print("\n🛑 Bot stopped by user")
        except Exception as e:
            print(f"❌ Critical error: {e}")
            logging.exception("Critical error in main")

if __name__ == "__main__":
    main()