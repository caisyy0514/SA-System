import ccxt.async_support as ccxt
import pandas as pd
import pandas_ta as ta
import asyncio
from typing import Dict, Any
import logging
from config import BotConfig

logger = logging.getLogger("sentinel.engine")

class DataEngine:
    def __init__(self, config: BotConfig):
        self.config = config
        # Ensure keys are stripped strings to avoid TypeError or whitespace issues
        api_key = str(config.OKX_API_KEY or "").strip()
        secret = str(config.OKX_SECRET_KEY or "").strip()
        passphrase = str(config.OKX_PASSPHRASE or "").strip()
        
        if not api_key or not secret or not passphrase:
            logger.warning("[交易所] 检测到 OKX 密钥不完整，初始化可能受限")

        self.exchange = ccxt.okx({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            'options': {'defaultType': 'swap'},  # Perpetual Swap
            'enableRateLimit': True,
        })
        if config.IS_SANDBOX:
            self.exchange.set_sandbox_mode(True)

    async def initialize(self):
        try:
            await self.exchange.load_markets()
            logger.info("[交易所] OKX 连接已初始化")
        except Exception as e:
            logger.error(f"[交易所] 连接 OKX 失败: {e}")
            raise e

    async def close(self):
        await self.exchange.close()

    async def fetch_market_snapshot(self, symbol: str) -> Dict[str, Any]:
        """
        Fetches comprehensive market data for the AI context.
        """
        try:
            # Parallel fetch for speed
            ohlcv_task = self.exchange.fetch_ohlcv(symbol, self.config.TIMEFRAME, limit=100)
            ticker_task = self.exchange.fetch_ticker(symbol)
            funding_task = self.exchange.fetch_funding_rate(symbol)
            ob_task = self.exchange.fetch_order_book(symbol, limit=20)

            ohlcv, ticker, funding, ob = await asyncio.gather(
                ohlcv_task, ticker_task, funding_task, ob_task
            )

            # Process OHLCV into DataFrame with Indicators
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Technical Indicators
            df['atr'] = df.ta.atr(length=14)
            df['rsi'] = df.ta.rsi(length=14)
            df['ema_20'] = df.ta.ema(length=20)
            df['ema_50'] = df.ta.ema(length=50)

            last_row = df.iloc[-1]
            prev_row = df.iloc[-2]

            return {
                "symbol": symbol,
                "price": ticker['last'],
                "bid": ob['bids'][0][0],
                "ask": ob['asks'][0][0],
                "spread_pct": (ob['asks'][0][0] - ob['bids'][0][0]) / ob['bids'][0][0],
                "funding_rate": funding['fundingRate'],
                "vol_24h": ticker['quoteVolume'],
                "technical": {
                    "rsi": last_row['rsi'],
                    "atr": last_row['atr'],
                    "ema_trend": "BULLISH" if last_row['ema_20'] > last_row['ema_50'] else "BEARISH",
                    "close": last_row['close'],
                    "prev_close": prev_row['close']
                },
                "orderbook_imbalance": self._calc_ob_imbalance(ob)
            }
        except Exception as e:
            logger.error(f"[交易所] 获取 {symbol} 数据出错: {e}")
            return None

    def _calc_ob_imbalance(self, ob) -> str:
        try:
            bid_vol = sum([x[1] for x in ob['bids'][:10]])
            ask_vol = sum([x[1] for x in ob['asks'][:10]])
            ratio = bid_vol / (ask_vol + 1e-9)
            if ratio > 1.5: return "STRONG_BUY_WALL"
            if ratio < 0.6: return "STRONG_SELL_WALL"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"

    async def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float) -> float:
        """
        Volatility Adjusted Position Sizing.
        """
        if entry_price <= 0 or stop_loss <= 0: return 0.0
        
        risk_per_unit = abs(entry_price - stop_loss)
        if risk_per_unit == 0: return 0.0

        amount = self.config.MAX_RISK_PER_TRADE_USD / risk_per_unit
        
        # Check min limits
        try:
            market = self.exchange.market(symbol)
            min_amount = market['limits']['amount']['min']
            if amount < min_amount:
                logger.warning(f"[交易所] 计算的仓位 {amount} 低于最小限制 {min_amount}")
                return 0.0
        except Exception:
            pass # Skip check if market info not available
            
        return amount

    async def execute_strategy(self, symbol: str, action: str, entry: float, sl: float, tp: float):
        try:
            amount = await self.calculate_position_size(symbol, entry, sl)
            if amount == 0: return

            side = 'buy' if action == 'LONG' else 'sell'
            
            # Set Leverage
            try:
                await self.exchange.set_leverage(self.config.LEVERAGE, symbol)
            except Exception:
                pass 

            params = {'stopLossPrice': sl, 'takeProfitPrice': tp}
            order = await self.exchange.create_order(symbol, 'market', side, amount, params=params)
            
            logger.info(f"[交易所] 订单已执行: {side} {amount} {symbol} @ 市价。止损: {sl}, 止盈: {tp}")
            return order

        except Exception as e:
            logger.error(f"[交易所] 执行失败: {e}")
            raise e
