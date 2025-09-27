"""
Exchange connector for multiple cryptocurrency exchanges
"""
import asyncio
import ccxt
import pandas as pd
import logging
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import time

from config import config

class ExchangeConnector:
    """
    Unified exchange connector supporting multiple exchanges
    Optimized for opportunistic trading with fast execution
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.exchange = None
        self.exchange_name = config.EXCHANGE.lower()
        self.connected = False
        self.rate_limiter = {}
        
    async def connect(self):
        """Connect to the specified exchange"""
        try:
            # Configure exchange
            exchange_config = {
                'apiKey': config.API_KEY,
                'secret': config.API_SECRET,
                'sandbox': config.SANDBOX,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'spot',  # Use spot trading
                    'createMarketBuyOrderRequiresPrice': False,
                }
            }
            
            # Initialize exchange
            if self.exchange_name == 'binance':
                self.exchange = ccxt.binance(exchange_config)
            elif self.exchange_name == 'bybit':
                self.exchange = ccxt.bybit(exchange_config)
            elif self.exchange_name == 'okx':
                self.exchange = ccxt.okx(exchange_config)
            else:
                self.exchange = ccxt.binance(exchange_config)  # Default to Binance
                
            # Test connection
            await self.exchange.load_markets()
            balance = await self.exchange.fetch_balance()
            
            self.connected = True
            self.logger.info(f"✅ Connected to {self.exchange.name}")
            
            # Log available balance
            if 'USDT' in balance['total']:
                usdt_balance = balance['total']['USDT']
                self.logger.info(f"💰 Available USDT balance: ${usdt_balance:.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to connect to {self.exchange_name}: {e}")
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from exchange"""
        if self.exchange:
            await self.exchange.close()
        self.connected = False
        self.logger.info(f"🔌 Disconnected from {self.exchange_name}")
    
    async def get_ohlcv(self, symbol: str, timeframe: str = '5m', limit: int = 200) -> Optional[pd.DataFrame]:
        """
        Get OHLCV data for a symbol
        """
        if not self.connected:
            return None
            
        try:
            # Rate limiting
            await self._rate_limit('ohlcv')
            
            # Fetch data
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            if not ohlcv:
                return None
                
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV for {symbol}: {e}")
            return None
    
    async def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        if not self.connected:
            return None
            
        try:
            await self._rate_limit('ticker')
            ticker = await self.exchange.fetch_ticker(symbol)
            return ticker['last']
            
        except Exception as e:
            self.logger.error(f"Error fetching price for {symbol}: {e}")
            return None
    
    async def get_balance(self) -> Dict:
        """Get account balance"""
        if not self.connected:
            return {}
            
        try:
            await self._rate_limit('balance')
            balance = await self.exchange.fetch_balance()
            return balance
            
        except Exception as e:
            self.logger.error(f"Error fetching balance: {e}")
            return {}
    
    async def create_order(self, symbol: str, type_: str, side: str, 
                          amount: float, price: float = None, params: Dict = None) -> Optional[Dict]:
        """
        Create trading order
        """
        if not self.connected:
            return None
            
        try:
            await self._rate_limit('trade')
            
            # Log order attempt
            self.logger.info(f"📝 Creating {side} {type_} order for {symbol}: {amount} @ {price or 'market'}")
            
            if config.SANDBOX:
                # Simulate order execution in sandbox mode
                order_result = {
                    'id': f"sim_{int(time.time())}",
                    'symbol': symbol,
                    'type': type_,
                    'side': side,
                    'amount': amount,
                    'price': price,
                    'status': 'filled',
                    'filled': amount,
                    'average': price or await self.get_current_price(symbol),
                    'cost': amount * (price or await self.get_current_price(symbol) or 0),
                    'timestamp': int(time.time() * 1000)
                }
                
                self.logger.info(f"🔄 Simulated order executed: {order_result['id']}")
                return order_result
            else:
                # Real order execution
                order_result = await self.exchange.create_order(
                    symbol, type_, side, amount, price, params or {}
                )
                
                self.logger.info(f"✅ Order executed: {order_result.get('id')}")
                return order_result
                
        except Exception as e:
            self.logger.error(f"❌ Failed to create order for {symbol}: {e}")
            return None
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel a specific order"""
        if not self.connected:
            return False
            
        try:
            await self._rate_limit('trade')
            
            if config.SANDBOX:
                self.logger.info(f"🔄 Simulated order cancellation: {order_id}")
                return True
            else:
                await self.exchange.cancel_order(order_id, symbol)
                self.logger.info(f"❌ Cancelled order: {order_id}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            return False
    
    async def cancel_all_orders(self, symbol: str = None) -> bool:
        """Cancel all orders for a symbol or all symbols"""
        if not self.connected:
            return False
            
        try:
            await self._rate_limit('trade')
            
            if config.SANDBOX:
                self.logger.info(f"🔄 Simulated cancellation of all orders for {symbol or 'all symbols'}")
                return True
            else:
                if symbol:
                    orders = await self.exchange.fetch_open_orders(symbol)
                    for order in orders:
                        await self.cancel_order(order['id'], symbol)
                else:
                    # Cancel all orders for all symbols
                    await self.exchange.cancel_all_orders()
                    
                self.logger.info(f"❌ Cancelled all orders for {symbol or 'all symbols'}")
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to cancel all orders: {e}")
            return False
    
    async def get_positions(self) -> List[Dict]:
        """Get current positions"""
        if not self.connected:
            return []
            
        try:
            await self._rate_limit('positions')
            
            if hasattr(self.exchange, 'fetch_positions'):
                positions = await self.exchange.fetch_positions()
                # Filter out zero positions
                return [pos for pos in positions if pos['contracts'] > 0]
            else:
                # For spot trading, calculate positions from balance
                balance = await self.get_balance()
                positions = []
                
                for currency, data in balance['total'].items():
                    if data > 0 and currency != 'USDT':
                        positions.append({
                            'symbol': f"{currency}/USDT",
                            'contracts': data,
                            'side': 'long',
                            'unrealized_pnl': 0,
                            'entry_price': 0
                        })
                        
                return positions
                
        except Exception as e:
            self.logger.error(f"Error fetching positions: {e}")
            return []
    
    async def get_order_book(self, symbol: str, limit: int = 20) -> Optional[Dict]:
        """Get order book data"""
        if not self.connected:
            return None
            
        try:
            await self._rate_limit('orderbook')
            order_book = await self.exchange.fetch_order_book(symbol, limit)
            return order_book
            
        except Exception as e:
            self.logger.error(f"Error fetching order book for {symbol}: {e}")
            return None
    
    async def get_recent_trades(self, symbol: str, limit: int = 50) -> List[Dict]:
        """Get recent trades for a symbol"""
        if not self.connected:
            return []
            
        try:
            await self._rate_limit('trades')
            trades = await self.exchange.fetch_trades(symbol, limit=limit)
            return trades
            
        except Exception as e:
            self.logger.error(f"Error fetching recent trades for {symbol}: {e}")
            return []
    
    async def get_funding_rates(self, symbols: List[str] = None) -> Dict:
        """Get funding rates for futures trading"""
        if not self.connected:
            return {}
            
        try:
            await self._rate_limit('funding')
            
            if hasattr(self.exchange, 'fetch_funding_rates'):
                if symbols:
                    rates = {}
                    for symbol in symbols:
                        rate = await self.exchange.fetch_funding_rate(symbol)
                        rates[symbol] = rate
                    return rates
                else:
                    return await self.exchange.fetch_funding_rates()
            else:
                return {}
                
        except Exception as e:
            self.logger.error(f"Error fetching funding rates: {e}")
            return {}
    
    async def _rate_limit(self, operation: str):
        """Simple rate limiting"""
        now = time.time()
        last_call = self.rate_limiter.get(operation, 0)
        
        # Different limits for different operations
        limits = {
            'ohlcv': 0.1,      # 10 requests per second
            'ticker': 0.05,    # 20 requests per second
            'balance': 1.0,    # 1 request per second
            'trade': 0.2,      # 5 requests per second
            'positions': 0.5,  # 2 requests per second
            'orderbook': 0.1,  # 10 requests per second
            'trades': 0.2,     # 5 requests per second
            'funding': 2.0,    # 1 request per 2 seconds
        }
        
        wait_time = limits.get(operation, 0.1)
        elapsed = now - last_call
        
        if elapsed < wait_time:
            sleep_time = wait_time - elapsed
            await asyncio.sleep(sleep_time)
        
        self.rate_limiter[operation] = time.time()
    
    def get_exchange_info(self) -> Dict:
        """Get exchange information"""
        if not self.connected:
            return {}
            
        return {
            'name': self.exchange.name if self.exchange else 'Not connected',
            'sandbox': config.SANDBOX,
            'rate_limit': self.exchange.rateLimit if self.exchange else 0,
            'markets': len(self.exchange.markets) if self.exchange and hasattr(self.exchange, 'markets') else 0,
            'connected': self.connected
        }
    
    async def test_connectivity(self) -> bool:
        """Test exchange connectivity"""
        try:
            if not self.connected:
                await self.connect()
                
            # Simple test - fetch server time
            if hasattr(self.exchange, 'fetch_time'):
                server_time = await self.exchange.fetch_time()
                local_time = int(time.time() * 1000)
                time_diff = abs(server_time - local_time)
                
                if time_diff > 5000:  # 5 second difference
                    self.logger.warning(f"⚠️ Large time difference with server: {time_diff}ms")
                
            self.logger.info("✅ Connectivity test passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Connectivity test failed: {e}")
            return False