import hashlib
import hmac
import json
import datetime
import requests
from typing import Dict, List, Optional, Set, Tuple
from urllib.parse import urlencode
from dotenv import load_dotenv
import os
import time
from dataclasses import dataclass
import numpy as np

# Load environment variables
load_dotenv()

@dataclass
class SymbolConfig:
    """Configuration for each trading symbol"""
    symbol: str
    timeframe: str = "5m"
    momentum_period: int = 20
    momentum_threshold: float = 2.0
    position_size_pct: float = 0.01
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0
    trailing_stop_pct: float = 1.0
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    volume_ma_period: int = 20
    active: bool = True

class MultiSymbolStrategy:
    def __init__(self):
        """Initialize the multi-symbol trading strategy"""
        self.api_key = os.getenv('DELTA_API_KEY')
        self.api_secret = os.getenv('DELTA_API_SECRET')
        if not self.api_key or not self.api_secret:
            raise ValueError("API key and secret must be set in .env file")
        
        self.base_url = "https://api.india.delta.exchange"
        self.symbols: Dict[str, SymbolConfig] = {}
        self.available_symbols: Set[str] = set()
        self._load_available_symbols()
        self._load_saved_symbols()

    def _load_available_symbols(self):
        """Load all available trading symbols from Delta Exchange"""
        products = self._make_request('GET', '/products')
        if products.get('success'):
            self.available_symbols = {p['symbol'] for p in products.get('result', [])}
            print(f"Available symbols: {len(self.available_symbols)}")

    def _save_symbols(self):
        """Save symbols configuration to a JSON file"""
        symbols_data = {
            symbol: {
                'timeframe': config.timeframe,
                'momentum_period': config.momentum_period,
                'momentum_threshold': config.momentum_threshold,
                'position_size_pct': config.position_size_pct,
                'stop_loss_pct': config.stop_loss_pct,
                'take_profit_pct': config.take_profit_pct,
                'trailing_stop_pct': config.trailing_stop_pct,
                'rsi_period': config.rsi_period,
                'rsi_overbought': config.rsi_overbought,
                'rsi_oversold': config.rsi_oversold,
                'volume_ma_period': config.volume_ma_period,
                'active': config.active
            }
            for symbol, config in self.symbols.items()
        }
        with open('symbols_config.json', 'w') as f:
            json.dump(symbols_data, f, indent=4)

    def _load_saved_symbols(self):
        """Load symbols configuration from JSON file"""
        try:
            if os.path.exists('symbols_config.json'):
                with open('symbols_config.json', 'r') as f:
                    symbols_data = json.load(f)
                for symbol, config in symbols_data.items():
                    self.symbols[symbol] = SymbolConfig(
                        symbol=symbol,
                        timeframe=config['timeframe'],
                        momentum_period=config['momentum_period'],
                        momentum_threshold=config['momentum_threshold'],
                        position_size_pct=config['position_size_pct'],
                        stop_loss_pct=config['stop_loss_pct'],
                        take_profit_pct=config['take_profit_pct'],
                        trailing_stop_pct=config['trailing_stop_pct'],
                        rsi_period=config['rsi_period'],
                        rsi_overbought=config['rsi_overbought'],
                        rsi_oversold=config['rsi_oversold'],
                        volume_ma_period=config['volume_ma_period'],
                        active=config['active']
                    )
        except Exception as e:
            print(f"Error loading saved symbols: {e}")

    def add_symbol(self, symbol: str, **kwargs) -> bool:
        """
        Add a new symbol to trade
        
        Parameters:
        - symbol: Trading pair (e.g., "BTCUSD")
        - **kwargs: Optional parameters for SymbolConfig
        
        Returns:
        - bool: True if symbol was added successfully
        """
        if symbol not in self.available_symbols:
            print(f"Warning: {symbol} is not available on Delta Exchange")
            return False
            
        self.symbols[symbol] = SymbolConfig(symbol=symbol, **kwargs)
        print(f"Added {symbol} to trading symbols")
        self._save_symbols()  # Save after adding
        return True

    def remove_symbol(self, symbol: str) -> bool:
        """
        Remove a symbol from trading
        
        Parameters:
        - symbol: Symbol to remove
        
        Returns:
        - bool: True if symbol was removed successfully
        """
        if symbol in self.symbols:
            del self.symbols[symbol]
            print(f"Removed {symbol} from trading symbols")
            self._save_symbols()  # Save after removing
            return True
        return False

    def activate_symbol(self, symbol: str) -> bool:
        """Activate trading for a symbol"""
        if symbol in self.symbols:
            self.symbols[symbol].active = True
            print(f"Activated trading for {symbol}")
            self._save_symbols()  # Save after activating
            return True
        return False

    def deactivate_symbol(self, symbol: str) -> bool:
        """Deactivate trading for a symbol"""
        if symbol in self.symbols:
            self.symbols[symbol].active = False
            print(f"Deactivated trading for {symbol}")
            self._save_symbols()  # Save after deactivating
            return True
        return False

    def list_symbols(self) -> List[Dict]:
        """List all configured symbols and their status"""
        return [
            {
                'symbol': symbol,
                'config': {
                    'timeframe': config.timeframe,
                    'momentum_period': config.momentum_period,
                    'momentum_threshold': config.momentum_threshold,
                    'position_size_pct': config.position_size_pct,
                    'stop_loss_pct': config.stop_loss_pct,
                    'take_profit_pct': config.take_profit_pct,
                    'trailing_stop_pct': config.trailing_stop_pct,
                    'rsi_period': config.rsi_period,
                    'rsi_overbought': config.rsi_overbought,
                    'rsi_oversold': config.rsi_oversold,
                    'volume_ma_period': config.volume_ma_period,
                    'active': config.active
                }
            }
            for symbol, config in self.symbols.items()
        ]

    def get_time_stamp(self):
        """Get current UTC timestamp in seconds"""
        d = datetime.datetime.utcnow()
        epoch = datetime.datetime(1970, 1, 1)
        return str(int((d - epoch).total_seconds()))

    def generate_signature(self, secret, message):
        """Generate HMAC SHA256 signature"""
        message = bytes(message, 'utf-8')
        secret = bytes(secret, 'utf-8')
        hash = hmac.new(secret, message, hashlib.sha256)
        return hash.hexdigest()

    def _make_request(self, method: str, path: str, params: Optional[Dict] = None, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to Delta Exchange API"""
        if not path.startswith('/v2'):
            path = '/v2' + path
            
        query_string = ""
        if params:
            sorted_params = dict(sorted(params.items()))
            query_string = urlencode(sorted_params)
        
        url = f"{self.base_url}{path}"
        if query_string:
            url = f"{url}?{query_string}"
            
        timestamp = self.get_time_stamp()
        
        body = ""
        if data:
            body = json.dumps(data, separators=(',', ':'))
            
        signature_path = path
        if query_string:
            signature_path = f"{path}?{query_string}"
            
        message = method.upper() + timestamp + signature_path + body
        signature = self.generate_signature(self.api_secret, message)
        
        headers = {
            'api-key': self.api_key,
            'timestamp': timestamp,
            'signature': signature,
            'Content-Type': 'application/json'
        }

        try:
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers)
            else:
                response = requests.request(method, url, headers=headers, data=body)
            return response.json()
            
        except Exception as e:
            print(f"API request failed: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_historical_data(self, symbol: str, resolution: str = '1m', start: Optional[str] = None, end: Optional[str] = None) -> Dict:
        """Get historical candle data"""
        if not start:
            start_time = datetime.datetime.utcnow() - datetime.timedelta(days=1)
            start = str(int((start_time - datetime.datetime(1970,1,1)).total_seconds()))
        
        if not end:
            end_time = datetime.datetime.utcnow()
            end = str(int((end_time - datetime.datetime(1970,1,1)).total_seconds()))
        
        params = {
            'symbol': symbol,
            'resolution': resolution,
            'start': start,
            'end': end
        }
        return self._make_request('GET', '/history/candles', params=params)

    def place_order(self, product_id: int, size: float, side: str, order_type: str = 'market_order', stop_price: Optional[float] = None, price: Optional[float] = None) -> Dict:
        """Place a new order"""
        data = {
            'product_id': product_id,
            'size': size,
            'side': side,
            'order_type': order_type
        }
        if stop_price:
            data['stop_price'] = stop_price
        if price:
            data['price'] = price
        return self._make_request('POST', '/orders', data=data)

    def calculate_rsi(self, prices: List[float], period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(prices) < period + 1:
            return 50.0  # Default to neutral if not enough data
            
        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_volatility_breakout(self, prices: List[float], periods: int = 20) -> Tuple[float, float, float]:
        """Calculate volatility breakout indicators
        Returns: (volatility, acceleration, breakout_strength)"""
        if len(prices) < periods:
            return 0.0, 0.0, 0.0

        # Calculate price changes
        returns = [((prices[i] - prices[i-1]) / prices[i-1]) * 100 for i in range(1, len(prices))]
        
        # Recent volatility (last 20 periods)
        recent_volatility = np.std(returns[-periods:]) if len(returns) >= periods else 0
        
        # Price acceleration (rate of change of returns)
        # Use shorter window (5 periods) for faster reaction
        recent_returns = returns[-5:]
        acceleration = sum(recent_returns) / len(recent_returns) if recent_returns else 0
        
        # Calculate short-term volatility (5 periods)
        short_volatility = np.std(returns[-5:]) if len(returns) >= 5 else 0
        
        # Breakout strength relative to short-term volatility
        latest_return = returns[-1] if returns else 0
        breakout_strength = latest_return / short_volatility if short_volatility > 0 else 0
        
        return recent_volatility, acceleration, breakout_strength

    def calculate_signal(self, symbol: str) -> Dict:
        """Calculate trading signal for a symbol"""
        config = self.symbols.get(symbol)
        if not config or not config.active:
            return {'success': False, 'error': 'Symbol not configured or inactive'}

        try:
            # Get historical data with more candles for better analysis
            history = self.get_historical_data(symbol, config.timeframe)
            if not history.get('success'):
                return {'success': False, 'error': f"Failed to get historical data: {history.get('error')}"}

            candles = history.get('result', [])
            if not candles:
                return {'success': False, 'error': "No historical data available"}

            # Calculate technical indicators
            close_prices = [float(candle['close']) for candle in candles]
            volumes = [float(candle.get('volume', 0)) for candle in candles]

            if len(close_prices) < max(config.momentum_period, config.rsi_period, config.volume_ma_period):
                return {'success': False, 'error': f"Not enough data for calculations"}

            # Calculate RSI
            rsi = self.calculate_rsi(close_prices, config.rsi_period)

            # Calculate volume moving average
            recent_volumes = volumes[-config.volume_ma_period:]
            volume_ma = sum(recent_volumes) / len(recent_volumes)
            current_volume = volumes[-1]
            volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0

            # Calculate volatility breakout indicators
            volatility, acceleration, breakout_strength = self.calculate_volatility_breakout(close_prices)

            # Generate signal based on breakout detection
            signal = 'neutral'
            
            # For buy signal:
            # 1. Strong positive acceleration or breakout
            # 2. Above average volume
            # 3. RSI not extremely overbought
            if ((acceleration > 0.5 or breakout_strength > 1.5) and 
                volume_ratio > 1.2 and
                rsi < 80):
                signal = 'buy'
                
            # For sell signal:
            # 1. Strong negative acceleration or breakout
            # 2. Above average volume
            # 3. RSI not extremely oversold
            elif ((acceleration < -0.5 or breakout_strength < -1.5) and
                  volume_ratio > 1.2 and
                  rsi > 20):
                signal = 'sell'

            return {
                'success': True,
                'symbol': symbol,
                'current_price': close_prices[-1],
                'acceleration': acceleration,
                'breakout_strength': breakout_strength,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'signal': signal
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}

    def execute_trade(self, symbol: str, signal: str, current_price: float, 
                     available_balance: float, trade_amount: float, 
                     acceleration: float, breakout_strength: float) -> Dict:
        """Execute a trade based on the signal"""
        if signal not in ['buy', 'sell']:
            return None

        # Calculate dynamic position size based on signal strength
        signal_strength = min(abs(acceleration) + abs(breakout_strength), 3.0) / 3.0
        adjusted_trade_amount = trade_amount * signal_strength

        # Calculate stop loss and take profit levels
        config = self.symbols[symbol]
        if signal == 'buy':
            entry_price = current_price
            stop_loss = entry_price * (1 - config.stop_loss_pct/100)
            take_profit = entry_price * (1 + config.take_profit_pct/100)
        else:  # sell
            entry_price = current_price
            stop_loss = entry_price * (1 + config.stop_loss_pct/100)
            take_profit = entry_price * (1 - config.take_profit_pct/100)

        return {
            'type': signal,
            'entry_price': entry_price,
            'size': adjusted_trade_amount,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_stop': config.trailing_stop_pct/100
        }

    def update_position(self, position: Dict, current_price: float) -> Tuple[Dict, float]:
        """Update position with trailing stop and check for exit conditions
        Returns: (updated_position, profit_loss)"""
        if not position:
            return None, 0.0

        profit_loss = 0.0
        exit_signal = False
        exit_price = current_price

        if position['type'] == 'buy':
            # Calculate profit/loss percentage
            price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            
            # Update trailing stop if in profit
            if price_change_pct > position['trailing_stop']:
                new_stop = current_price * (1 - position['trailing_stop'])
                position['stop_loss'] = max(position['stop_loss'], new_stop)

            # Check exit conditions
            if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                exit_signal = True
                
        else:  # sell position
            # Calculate profit/loss percentage
            price_change_pct = (position['entry_price'] - current_price) / position['entry_price']
            
            # Update trailing stop if in profit
            if price_change_pct > position['trailing_stop']:
                new_stop = current_price * (1 + position['trailing_stop'])
                position['stop_loss'] = min(position['stop_loss'], new_stop)

            # Check exit conditions
            if current_price >= position['stop_loss'] or current_price <= position['take_profit']:
                exit_signal = True

        # Calculate profit/loss if exiting
        if exit_signal:
            if position['type'] == 'buy':
                profit_loss = (exit_price - position['entry_price']) * position['size']
            else:
                profit_loss = (position['entry_price'] - exit_price) * position['size']
            return None, profit_loss

        return position, profit_loss

    def execute_strategy(self, backtest: bool = False) -> List[Dict]:
        """
        Execute the trading strategy for all active symbols
        
        Parameters:
        - backtest: If True, only simulate trades without placing real orders
        
        Returns:
        - List of execution results for each symbol
        """
        results = []
        
        for symbol, config in self.symbols.items():
            if not config.active:
                continue

            try:
                # Calculate signals
                signal_data = self.calculate_signal(symbol)
                if not signal_data['success']:
                    results.append({
                        'symbol': symbol,
                        'success': False,
                        'error': signal_data['error']
                    })
                    continue

                # Get wallet balance
                balance = self._make_request('GET', '/wallet/balances')
                available_balance = float(balance.get('result', [{}])[0].get('available_balance', 0))

                # Calculate position size
                trade_amount = available_balance * config.position_size_pct

                # Execute trade
                trade = self.execute_trade(
                    symbol=symbol,
                    signal=signal_data['signal'],
                    current_price=signal_data['current_price'],
                    available_balance=available_balance,
                    trade_amount=trade_amount,
                    acceleration=signal_data['acceleration'],
                    breakout_strength=signal_data['breakout_strength']
                )

                if trade:
                    # Update position
                    updated_position, profit_loss = self.update_position(trade, signal_data['current_price'])

                    result = {
                        'symbol': symbol,
                        'current_price': signal_data['current_price'],
                        'acceleration': signal_data['acceleration'],
                        'breakout_strength': signal_data['breakout_strength'],
                        'volatility': signal_data['volatility'],
                        'volume_ratio': signal_data['volume_ratio'],
                        'signal': signal_data['signal'],
                        'available_balance': available_balance,
                        'trade_amount': trade_amount,
                        'position': updated_position,
                        'profit_loss': profit_loss,
                        'mode': 'backtest' if backtest else 'live'
                    }

                    if not backtest and signal_data['signal'] in ['buy', 'sell']:
                        # Get product details
                        products = self._make_request('GET', '/products')
                        product = next((p for p in products.get('result', []) if p['symbol'] == symbol), None)
                        
                        if product:
                            # Place the main order
                            main_order = self.place_order(
                                product_id=product['id'],
                                size=trade['size'],
                                side=signal_data['signal'],
                                order_type='market_order'
                            )
                            result['main_order'] = main_order

                            # Place stop loss order
                            stop_order = self.place_order(
                                product_id=product['id'],
                                size=trade['size'],
                                side='sell' if signal_data['signal'] == 'buy' else 'buy',
                                order_type='stop_order',
                                stop_price=trade['stop_loss']
                            )
                            result['stop_order'] = stop_order

                            # Place take profit order
                            take_profit_order = self.place_order(
                                product_id=product['id'],
                                size=trade['size'],
                                side='sell' if signal_data['signal'] == 'buy' else 'buy',
                                order_type='limit_order',
                                price=trade['take_profit']
                            )
                            result['take_profit_order'] = take_profit_order
                    results.append(result)

            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'success': False,
                    'error': str(e)
                })

        return results

    def run_backtest(self, days=30):
        """Run strategy in backtest mode"""
        print(f"\nRunning backtest over the last {days} days...")
        
        # Initialize backtest metrics
        total_trades = 0
        winning_trades = 0
        total_profit_loss = 0
        max_drawdown = 0
        current_drawdown = 0
        peak_balance = self.initial_balance = 10000  # Start with 10,000 USD virtual balance
        
        # Get historical data for each symbol
        end_time = datetime.datetime.utcnow()
        start_time = end_time - datetime.timedelta(days=days)
        
        results = []
        for symbol in self.symbols:
            if not self.symbols[symbol].active:
                continue
                
            print(f"\nAnalyzing {symbol}...")
            historical_data = self.get_historical_data(
                symbol,
                resolution=self.symbols[symbol].timeframe,
                start=str(int((start_time - datetime.datetime(1970,1,1)).total_seconds())),
                end=str(int((end_time - datetime.datetime(1970,1,1)).total_seconds()))
            )
            
            if not historical_data.get('success'):
                print(f"Failed to get historical data for {symbol}")
                continue
                
            candles = historical_data.get('result', [])
            if not candles:
                print(f"No historical data available for {symbol}")
                continue
                
            # Calculate signals for each candle
            balance = self.initial_balance
            position = None
            
            for i in range(len(candles)):
                if i < self.symbols[symbol].momentum_period:
                    continue
                    
                # Calculate momentum
                current_price = float(candles[i]['close'])
                past_price = float(candles[i-self.symbols[symbol].momentum_period]['close'])
                momentum = ((current_price - past_price) / past_price) * 100
                
                # Calculate RSI
                close_prices = [float(candle['close']) for candle in candles[:i+1]]
                rsi = self.calculate_rsi(close_prices, self.symbols[symbol].rsi_period)
                
                # Calculate volume moving average
                volumes = [float(candle.get('volume', 0)) for candle in candles[:i+1]]
                recent_volumes = volumes[-self.symbols[symbol].volume_ma_period:]
                volume_ma = sum(recent_volumes) / len(recent_volumes)
                current_volume = volumes[-1]
                volume_ratio = current_volume / volume_ma if volume_ma > 0 else 1.0
                
                # Calculate volatility breakout indicators
                volatility, acceleration, breakout_strength = self.calculate_volatility_breakout(close_prices)

                # Generate signal with multiple confirmations
                signal = "neutral"
                
                # For buy signal:
                # 1. Strong positive acceleration or breakout
                # 2. Above average volume
                # 3. RSI not extremely overbought
                if ((acceleration > 0.5 or breakout_strength > 1.5) and 
                    volume_ratio > 1.2 and
                    rsi < 80):
                    signal = "buy"
                # For sell signal:
                # 1. Strong negative acceleration or breakout
                # 2. Above average volume
                # 3. RSI not extremely oversold
                elif ((acceleration < -0.5 or breakout_strength < -1.5) and
                      volume_ratio > 1.2 and
                      rsi > 20):
                    signal = "sell"
                
                # Execute trades
                if signal != "neutral" and position is None:
                    # Open position
                    position_size = balance * self.symbols[symbol].position_size_pct
                    position = {
                        'type': signal,
                        'entry_price': current_price,
                        'size': position_size / current_price
                    }
                    total_trades += 1
                    
                elif signal != "neutral" and position is not None:
                    # Check for exit conditions
                    if (position['type'] == 'buy' and acceleration < 0) or \
                       (position['type'] == 'sell' and acceleration > 0):
                        # Close position
                        position = None
            
            results.append({
                'symbol': symbol,
                'current_price': current_price,
                'acceleration': acceleration,
                'breakout_strength': breakout_strength,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'signal': signal,
                'available_balance': balance,
                'trade_amount': balance * self.symbols[symbol].position_size_pct,
                'mode': 'backtest'
            })
        
        # Print backtest summary
        print("\n=== Backtest Summary ===")
        print(f"Period: {days} days")
        print(f"Initial Balance: ${self.initial_balance:,.2f}")
        print(f"Final Balance: ${balance:,.2f}")
        print(f"Total Return: {((balance/self.initial_balance)-1)*100:.2f}%")
        print(f"Total Trades: {total_trades}")
        win_rate = (winning_trades/total_trades*100) if total_trades > 0 else 0
        print(f"Win Rate: {win_rate:.2f}%")
        print(f"Maximum Drawdown: {max_drawdown:.2f}%")
        
        return results

    def run_paper_trading(self):
        """Run paper trading simulation"""
        print("\nStarting paper trading simulation...")
        print("Initial balance: $10,000.00")
        
        # Initialize paper trading state
        self.paper_balance = 10000.0
        self.paper_positions = {}
        self.trade_history = []
        self.total_trades = 0
        self.winning_trades = 0
        
        try:
            while True:
                for symbol in self.symbols:
                    if not self.symbols[symbol].active:
                        continue

                    # Get trading signal
                    signal_data = self.calculate_signal(symbol)
                    if not signal_data['success']:
                        print(f"Error calculating signal for {symbol}: {signal_data.get('error')}")
                        continue

                    current_price = signal_data['current_price']
                    signal = signal_data['signal']
                    
                    # Position management
                    if symbol in self.paper_positions:
                        position = self.paper_positions[symbol]
                        updated_position, profit_loss = self.update_position(position, current_price)
                        
                        if updated_position is None:  # Position closed
                            self.paper_balance += profit_loss
                            self.total_trades += 1
                            if profit_loss > 0:
                                self.winning_trades += 1
                            
                            # Record trade
                            self.trade_history.append({
                                'symbol': symbol,
                                'type': position['type'],
                                'entry_price': position['entry_price'],
                                'exit_price': current_price,
                                'profit_loss': profit_loss,
                                'balance': self.paper_balance
                            })
                            
                            print(f"\nPosition closed - {symbol}")
                            print(f"Entry: ${position['entry_price']:.2f}")
                            print(f"Exit: ${current_price:.2f}")
                            print(f"P/L: ${profit_loss:.2f}")
                            print(f"Balance: ${self.paper_balance:.2f}")
                            
                            del self.paper_positions[symbol]
                        else:
                            self.paper_positions[symbol] = updated_position
                    
                    # New position entry
                    elif signal in ['buy', 'sell']:
                        trade_amount = self.paper_balance * self.symbols[symbol].position_size_pct / 100
                        
                        # Execute trade
                        trade = self.execute_trade(
                            symbol=symbol,
                            signal=signal,
                            current_price=current_price,
                            available_balance=self.paper_balance,
                            trade_amount=trade_amount,
                            acceleration=signal_data['acceleration'],
                            breakout_strength=signal_data['breakout_strength']
                        )
                        
                        if trade:
                            self.paper_positions[symbol] = trade
                            print(f"\nNew {signal.upper()} position - {symbol}")
                            print(f"Entry: ${current_price:.2f}")
                            print(f"Size: {trade['size']:.4f}")
                            print(f"Stop Loss: ${trade['stop_loss']:.2f}")
                            print(f"Take Profit: ${trade['take_profit']:.2f}")
                    
                    # Print status
                    print(f"\r{symbol}: ${current_price:.2f} | "
                          f"Acceleration: {signal_data['acceleration']:.2f}% | "
                          f"Breakout Strength: {signal_data['breakout_strength']:.2f} | "
                          f"Signal: {signal}", end='')
                    
                    if self.total_trades > 0:
                        win_rate = (self.winning_trades / self.total_trades) * 100
                        print(f"\nTotal Trades: {self.total_trades} | "
                              f"Win Rate: {win_rate:.1f}% | "
                              f"Current Balance: ${self.paper_balance:.2f}")
                    
                    time.sleep(1)  # Avoid hitting rate limits
                    
        except KeyboardInterrupt:
            print("\n\nPaper trading stopped.")
            print(f"Final Balance: ${self.paper_balance:.2f}")
            print(f"Total Return: {((self.paper_balance - 10000) / 10000 * 100):.2f}%")
            print(f"Total Trades: {self.total_trades}")
            if self.total_trades > 0:
                print(f"Win Rate: {(self.winning_trades / self.total_trades * 100):.2f}%")
            
            # Print trade history
            print("\nTrade History:")
            for trade in self.trade_history:
                print(f"{trade['symbol']} {trade['type'].upper()}: "
                      f"Entry ${trade['entry_price']:.2f} -> "
                      f"Exit ${trade['exit_price']:.2f} | "
                      f"P/L ${trade['profit_loss']:.2f}")

def run_multi_symbol_strategy():
    """Interactive menu for managing and running MultiSymbolStrategy"""
    try:
        # Initialize strategy
        strategy = MultiSymbolStrategy()
        
        while True:
            print("\n=== Delta Exchange Trading Bot ===")
            print("1. Add new symbol")
            print("2. Remove symbol")
            print("3. List all symbols")
            print("4. Activate symbol")
            print("5. Deactivate symbol")
            print("6. Run strategy (Backtest)")
            print("7. Run strategy (Live)")
            print("8. Run backtest")
            print("9. Run paper trading")
            print("10. Exit")
            
            choice = input("\nEnter your choice (1-10): ")
            
            if choice == "1":
                symbol = input("Enter symbol (e.g., BTCUSD): ").upper()
                print("\nTimeframe options: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
                timeframe = input("Enter timeframe (default: 5m): ") or "5m"
                momentum_period = int(input("Enter momentum period (default: 20): ") or "20")
                momentum_threshold = float(input("Enter momentum threshold % (default: 2.0): ") or "2.0")
                position_size = float(input("Enter position size % (default: 1.0): ") or "1.0") / 100
                stop_loss = float(input("Enter stop loss % (default: 2.0): ") or "2.0")
                take_profit = float(input("Enter take profit % (default: 4.0): ") or "4.0")
                trailing_stop = float(input("Enter trailing stop % (default: 1.0): ") or "1.0")
                rsi_period = int(input("Enter RSI period (default: 14): ") or "14")
                rsi_overbought = float(input("Enter RSI overbought level (default: 70.0): ") or "70.0")
                rsi_oversold = float(input("Enter RSI oversold level (default: 30.0): ") or "30.0")
                volume_ma_period = int(input("Enter volume MA period (default: 20): ") or "20")
                
                if strategy.add_symbol(symbol, 
                                     timeframe=timeframe,
                                     momentum_period=momentum_period,
                                     momentum_threshold=momentum_threshold,
                                     position_size_pct=position_size,
                                     stop_loss_pct=stop_loss,
                                     take_profit_pct=take_profit,
                                     trailing_stop_pct=trailing_stop,
                                     rsi_period=rsi_period,
                                     rsi_overbought=rsi_overbought,
                                     rsi_oversold=rsi_oversold,
                                     volume_ma_period=volume_ma_period):
                    print(f"\nSuccessfully added {symbol}")
                else:
                    print(f"\nFailed to add {symbol}. Make sure it's a valid symbol.")
            
            elif choice == "2":
                symbols = strategy.list_symbols()
                if not symbols:
                    print("\nNo symbols configured.")
                    continue
                    
                print("\nConfigured symbols:")
                for i, sym in enumerate(symbols, 1):
                    print(f"{i}. {sym['symbol']}")
                
                try:
                    index = int(input("\nEnter number to remove: ")) - 1
                    symbol = symbols[index]['symbol']
                    if strategy.remove_symbol(symbol):
                        print(f"\nSuccessfully removed {symbol}")
                    else:
                        print(f"\nFailed to remove {symbol}")
                except (ValueError, IndexError):
                    print("\nInvalid selection")
            
            elif choice == "3":
                symbols = strategy.list_symbols()
                if not symbols:
                    print("\nNo symbols configured.")
                    continue
                
                print("\nConfigured symbols:")
                print(json.dumps(symbols, indent=2))
            
            elif choice == "4":
                symbols = [s['symbol'] for s in strategy.list_symbols() if not s['config']['active']]
                if not symbols:
                    print("\nNo inactive symbols found.")
                    continue
                
                print("\nInactive symbols:")
                for i, symbol in enumerate(symbols, 1):
                    print(f"{i}. {symbol}")
                
                try:
                    index = int(input("\nEnter number to activate: ")) - 1
                    symbol = symbols[index]
                    if strategy.activate_symbol(symbol):
                        print(f"\nSuccessfully activated {symbol}")
                    else:
                        print(f"\nFailed to activate {symbol}")
                except (ValueError, IndexError):
                    print("\nInvalid selection")
            
            elif choice == "5":
                symbols = [s['symbol'] for s in strategy.list_symbols() if s['config']['active']]
                if not symbols:
                    print("\nNo active symbols found.")
                    continue
                
                print("\nActive symbols:")
                for i, symbol in enumerate(symbols, 1):
                    print(f"{i}. {symbol}")
                
                try:
                    index = int(input("\nEnter number to deactivate: ")) - 1
                    symbol = symbols[index]
                    if strategy.deactivate_symbol(symbol):
                        print(f"\nSuccessfully deactivated {symbol}")
                    else:
                        print(f"\nFailed to deactivate {symbol}")
                except (ValueError, IndexError):
                    print("\nInvalid selection")
            
            elif choice == "6":
                if not strategy.list_symbols():
                    print("\nNo symbols configured. Please add symbols first.")
                    continue
                
                print("\nRunning strategy (Backtest mode)...")
                results = strategy.execute_strategy(backtest=True)
                print("\nBacktest Results:")
                print(json.dumps(results, indent=2))
            
            elif choice == "7":
                if not strategy.list_symbols():
                    print("\nNo symbols configured. Please add symbols first.")
                    continue
                
                confirm = input("\nWARNING: This will execute live trades. Continue? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("\nLive trading cancelled")
                    continue
                
                print("\nRunning strategy (Live mode)...")
                results = strategy.execute_strategy(backtest=False)
                print("\nLive Trading Results:")
                print(json.dumps(results, indent=2))
            
            elif choice == "8":
                if not strategy.list_symbols():
                    print("\nNo symbols configured. Please add symbols first.")
                    continue
                
                days = int(input("\nEnter number of days for backtest (default: 30): ") or "30")
                results = strategy.run_backtest(days=days)
                print("\nBacktest Results:")
                print(json.dumps(results, indent=2))
            
            elif choice == "9":
                if not strategy.list_symbols():
                    print("\nNo symbols configured. Please add symbols first.")
                    continue
                
                strategy.run_paper_trading()
            
            elif choice == "10":
                print("\nExiting...")
                break
            
            else:
                print("\nInvalid choice. Please try again.")

    except Exception as e:
        print(f"Strategy execution error: {str(e)}")

if __name__ == "__main__":
    run_multi_symbol_strategy()
