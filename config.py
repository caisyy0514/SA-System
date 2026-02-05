from typing import List
from pydantic import BaseModel

class BotConfig(BaseModel):
    # --- Exchange Settings (OKX) ---
    OKX_API_KEY: str = ""
    OKX_SECRET_KEY: str = ""
    OKX_PASSPHRASE: str = ""
    IS_SANDBOX: bool = False
    
    # --- Trading Parameters ---
    TRADING_SYMBOLS: List[str] = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
    TIMEFRAME: str = "15m"
    LEVERAGE: int = 5
    MAX_RISK_PER_TRADE_USD: float = 50.0
    MIN_EXPECTED_VALUE: float = 1.5

    # --- AI Models API ---
    GEMINI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    
    # --- Monitoring ---
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""

    # --- System ---
    UPDATE_INTERVAL_SECONDS: int = 300
    WEB_PASSWORD: str = "admin" # Simple security for the web UI
