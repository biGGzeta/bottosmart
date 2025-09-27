# BottoSmart v2.0 - Intelligent Opportunistic Trading Bot 🚀

BottoSmart is an advanced, intelligent cryptocurrency trading bot designed with an **opportunistic and adaptive approach** to maximize trading opportunities in volatile markets. Unlike conservative bots, BottoSmart is built to be aggressive, adaptive, and intelligent in its trading decisions.

## 🎯 Key Features

### Opportunistic Trading Strategy
- **Aggressive Position Sizing**: Up to 15% of portfolio per trade for high-confidence opportunities
- **Multiple Opportunity Detection**: Momentum breakouts, reversal bounces, volatility expansion, trend continuation
- **Quick Execution**: 5-second cycle time for rapid opportunity capture
- **Dynamic Risk-Reward**: Targets 1.5:1+ risk-reward ratios with adaptive stop-losses

### Advanced Intelligence
- **Machine Learning Predictions**: Random Forest and Gradient Boosting models for market movement prediction
- **Technical Analysis**: 30+ technical indicators including RSI, MACD, Bollinger Bands, support/resistance
- **Market Structure Analysis**: Candlestick patterns, pivot points, market trend strength assessment
- **Volume Analysis**: Price-volume relationships and VWAP analysis

### Adaptive Behavior
- **Performance-Based Adjustment**: Automatically adjusts position sizes and confidence thresholds based on recent performance
- **Strategy Evolution**: Learns from winning and losing trades to optimize future decisions
- **Market Condition Recognition**: Adapts to different market conditions (trending, ranging, volatile)
- **Real-Time Risk Management**: Dynamic position sizing based on portfolio performance

### Comprehensive Risk Management
- **Multi-Layer Protection**: Stop-losses, take-profits, daily loss limits, maximum drawdown protection
- **Portfolio Monitoring**: Real-time portfolio exposure and risk metrics
- **Emergency Stops**: Automatic trading halt on severe adverse conditions
- **Volatility-Adjusted Sizing**: Position sizes adapt to market volatility

## 📁 Project Structure

```
bottosmart/
├── main.py                 # Main entry point and CLI
├── bottosmart.py          # Core trading bot engine
├── config.py              # Configuration management
├── market_analyzer.py     # Market analysis and opportunity detection
├── ml_predictor.py        # Machine learning prediction models
├── risk_manager.py        # Advanced risk management
├── exchange_connector.py  # Multi-exchange connectivity
├── requirements.txt       # Python dependencies
├── .env.example          # Environment configuration template
└── models/               # ML model storage (auto-created)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/biGGzeta/bottosmart.git
cd bottosmart

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (required for technical analysis)
# On Ubuntu/Debian:
sudo apt-get install ta-lib
# On macOS:
brew install ta-lib
# On Windows: Download from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
nano .env
```

Configure your API keys:
```env
EXCHANGE=binance
API_KEY=your_api_key_here
API_SECRET=your_api_secret_here
SANDBOX=true  # Start in sandbox mode
```

### 3. Test Components

```bash
# Test all components
python main.py --mode test

# Train ML models (optional, happens automatically)
python main.py --mode train
```

### 4. Run the Bot

```bash
# Start in sandbox mode (recommended first)
python main.py --sandbox

# Run in live mode (real money - be careful!)
python main.py
```

## 🎛️ Configuration Options

### Opportunistic Trading Parameters
- `MAX_POSITION_SIZE`: Maximum position size (default: 15% - aggressive)
- `MIN_POSITION_SIZE`: Minimum position size (default: 2%)
- `RISK_REWARD_RATIO`: Minimum risk/reward ratio (default: 1.5)
- `STOP_LOSS`: Stop loss percentage (default: 2.5% - tight for quick exits)
- `TAKE_PROFIT`: Take profit target (default: 6% - aggressive target)

### Adaptive Parameters
- `VOLATILITY_THRESHOLD`: Volatility threshold for opportunity detection
- `MOMENTUM_THRESHOLD`: Momentum threshold for breakout detection
- `PREDICTION_CONFIDENCE_THRESHOLD`: ML prediction confidence threshold (default: 65%)

### Risk Management
- `MAX_DAILY_LOSS`: Maximum daily portfolio loss (default: 5%)
- `MAX_DRAWDOWN`: Maximum portfolio drawdown (default: 15%)

## 🧠 Intelligence Features

### Machine Learning Models
- **Random Forest**: Ensemble learning for robust predictions
- **Gradient Boosting**: Sequential learning for pattern recognition
- **Feature Engineering**: 40+ features including technical indicators, market structure, volume analysis
- **Adaptive Training**: Models retrain automatically when performance degrades

### Opportunity Detection
1. **Momentum Breakouts**: High RSI + MACD + Bollinger Band breakouts
2. **Reversal Bounces**: Oversold conditions at support levels
3. **Volatility Expansion**: High volatility with strong trend
4. **Trend Continuation**: Sustained directional moves

### Market Analysis
- **Multi-Timeframe Analysis**: 5-minute charts with various indicators
- **Support/Resistance**: Dynamic pivot point calculation
- **Volume Profile**: Price-volume relationship analysis
- **Market Sentiment**: Composite sentiment scoring

## 📊 Performance Tracking

The bot tracks comprehensive performance metrics:
- **Win Rate**: Percentage of profitable trades
- **Average Return**: Mean return per trade
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest portfolio decline
- **Volatility**: Portfolio volatility metrics
- **Opportunity Success**: Percentage of detected opportunities executed

## ⚙️ Command Line Options

```bash
# Run modes
python main.py --mode run     # Normal trading mode
python main.py --mode test    # Test components
python main.py --mode train   # Train ML models

# Other options
python main.py --sandbox      # Force sandbox mode
python main.py --log-level DEBUG  # Detailed logging
```

## 🔒 Security & Risk Warnings

### ⚠️ Important Risk Disclaimers
- **High Risk**: This bot uses aggressive, opportunistic strategies
- **Real Money**: Live trading involves real financial risk
- **No Guarantees**: Past performance doesn't guarantee future results
- **Test First**: Always test in sandbox mode before live trading
- **Monitor Actively**: Don't leave the bot unattended for extended periods

### Security Best Practices
- Use API keys with trading permissions only (no withdrawals)
- Start with small amounts in sandbox mode
- Monitor the bot's performance closely
- Set appropriate risk limits for your risk tolerance
- Keep your API keys secure and never share them

## 🔧 Supported Exchanges

- **Binance** (Primary support)
- **Bybit** (Futures and Spot)
- **OKX** (Comprehensive support)
- **More exchanges**: Easy to add via CCXT library

## 📈 Example Performance

*Note: These are hypothetical examples based on backtesting. Real performance may vary significantly.*

- **Aggressive Mode**: 15% position sizes, 65% confidence threshold
- **Conservative Mode**: 8% position sizes, 75% confidence threshold
- **Adaptive Mode**: Dynamic sizing based on recent performance

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional exchange integrations
- New opportunity detection strategies
- Enhanced ML models
- Better risk management algorithms
- UI/Dashboard development

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading cryptocurrencies involves substantial risk and is not suitable for all investors. The high degree of leverage can work against you as well as for you. Before deciding to trade cryptocurrencies, you should carefully consider your investment objectives, level of experience, and risk appetite. You should be aware of all the risks associated with trading and seek advice from an independent financial advisor if you have any doubts.

---

**Made with 💜 by the BottoSmart team**

*"Be opportunistic, be intelligent, be profitable"*