
import asyncio
import logging
import httpx
import aiofiles
from datetime import datetime
from collections import deque
from typing import List, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from engine import DataEngine
from logic import DebateManager, TacticalPlan
from config import BotConfig

# --- Logging Setup with Buffer ---
log_buffer = deque(maxlen=200) # Keep last 200 logs

class BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            log_buffer.append(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
        except Exception:
            self.handleError(record)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("sentinel")
logger.addHandler(BufferHandler())

# --- Global State ---
class SystemState:
    running: bool = False
    task: asyncio.Task = None
    config: BotConfig = None
    engine: DataEngine = None
    brain: DebateManager = None
    signals: deque = deque(maxlen=50) # History of AI decisions
    current_symbol: Optional[str] = None # For "Analyzing" animation

state = SystemState()

# --- Helpers ---
async def telegram_alert(message: str, token: str, chat_id: str):
    if not token or not chat_id: return
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    async with httpx.AsyncClient() as client:
        try:
            await client.post(url, json={"chat_id": chat_id, "text": message})
        except Exception as e:
            logger.error(f"Telegram fail: {e}")

async def strategy_loop():
    logger.info(">>> 策略引擎循环已启动 <<<")
    
    while state.running:
        try:
            if not state.engine or not state.brain:
                logger.error("引擎或大脑未初始化。正在停止。")
                state.running = False
                break

            for symbol in state.config.TRADING_SYMBOLS:
                if not state.running: break
                
                state.current_symbol = symbol
                logger.info(f"[系统] 正在分析 {symbol}...")
                
                # 1. 获取数据
                data = await state.engine.fetch_market_snapshot(symbol)
                if not data: 
                    logger.warning(f"[系统] 无法获取 {symbol} 的数据")
                    continue

                # 2. AI 对抗决策
                plan = await state.brain.generate_tactics(data)
                
                # 存储历史
                state.signals.appendleft(plan)
                
                # 3. 报告结果
                log_msg = f"[系统] {symbol} 分析完成: {plan.action} | 期望值: {plan.expected_value} | 是否执行: {plan.should_trade}"
                logger.info(log_msg)
                
                if plan.should_trade:
                    logger.info(f"[系统] 🚨 正在触发 {symbol} 的执行指令 🚨")
                    await telegram_alert(
                        f"🚨 信号触发 {symbol} 🚨\n动作: {plan.action}\n入场: {plan.entry}\n理由: {plan.rationale}", 
                        state.config.TELEGRAM_BOT_TOKEN, state.config.TELEGRAM_CHAT_ID
                    )
                    
                    # 4. 执行交易
                    try:
                        await state.engine.execute_strategy(
                            symbol, plan.action, plan.entry, plan.stop_loss, plan.take_profit
                        )
                    except Exception as e:
                        logger.error(f"[系统] {symbol} 执行失败: {e}")
            
            state.current_symbol = None
            # 等待下一周期
            await asyncio.sleep(state.config.UPDATE_INTERVAL_SECONDS)
            
        except asyncio.CancelledError:
            logger.info("循环已取消。")
            break
        except Exception as e:
            logger.error(f"循环崩溃: {e}")
            state.current_symbol = None
            await asyncio.sleep(60)

# --- API ---

app = FastAPI(title="Sentinel-Adversary")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    try:
        async with aiofiles.open("index.html", mode="r", encoding="utf-8") as f:
            content = await f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Error: index.html not found</h1>", status_code=404)

@app.get("/api/status")
async def get_status():
    return {
        "running": state.running,
        "logs": list(log_buffer),
        "signals": list(state.signals),
        "current_symbol": state.current_symbol,
        "uptime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

@app.post("/api/start")
async def start_bot(config: BotConfig):
    if state.running:
        return {"message": "Already running"}
    
    state.config = config
    
    try:
        logger.info("[系统] 正在初始化数据引擎...")
        state.engine = DataEngine(config)
        await state.engine.initialize()
        
        logger.info("[系统] 正在初始化大脑...")
        state.brain = DebateManager(config)
        
    except Exception as e:
        logger.error(f"[系统] 初始化失败: {e}")
        raise HTTPException(status_code=400, detail=f"初始化失败: {str(e)}")

    state.running = True
    state.task = asyncio.create_task(strategy_loop())
    
    return {"message": "哨兵系统已启动"}

@app.post("/api/stop")
async def stop_bot():
    if not state.running:
        return {"message": "未在运行"}
    
    state.running = False
    if state.task:
        state.task.cancel()
        try:
            await state.task
        except asyncio.CancelledError:
            pass
            
    if state.engine:
        await state.engine.close()
        
    logger.info(">>> 系统已停止 <<<")
    return {"message": "哨兵系统已停止"}
