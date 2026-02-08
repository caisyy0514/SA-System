
import json
import logging
from pydantic import BaseModel
from typing import Optional, Literal, List
from google import genai 
from openai import AsyncOpenAI
from config import BotConfig

logger = logging.getLogger("sentinel.logic")

# --- 结构化输出定义 ---

class StrategyProposal(BaseModel):
    symbol: str
    action: Literal["LONG", "SHORT", "WAIT"]
    entry_price: float
    stop_loss: float
    take_profit: float
    reasoning: str
    confidence_score: int  # 0-100

class AuditReport(BaseModel):
    approved: bool
    risk_flags: List[str]
    liquidity_check: str
    revised_confidence: int # 0-100
    auditor_comment: str

class TacticalPlan(BaseModel):
    timestamp: str = ""
    should_trade: bool
    symbol: str
    action: str
    entry: float
    stop_loss: float
    take_profit: float
    expected_value: float
    rationale: str
    proposal: Optional[StrategyProposal] = None
    audit: Optional[AuditReport] = None

# --- 对抗决策逻辑 ---

class DebateManager:
    def __init__(self, config: BotConfig):
        self.config = config
        self.client_a = None
        self.client_b = None

        # 模型 A: Gemini (策略师)
        if config.GEMINI_API_KEY:
            try:
                self.client_a = genai.Client(api_key=config.GEMINI_API_KEY)
            except Exception as e:
                logger.error(f"Gemini 初始化错误: {e}")
        
        # 模型 B: DeepSeek (审计员)
        if config.DEEPSEEK_API_KEY:
            self.client_b = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY, 
                base_url="https://api.deepseek.com/v1"
            )

    async def _call_model_a(self, market_data: dict) -> StrategyProposal:
        if not self.client_a:
             return StrategyProposal(symbol=market_data['symbol'], action="WAIT", entry_price=0, stop_loss=0, take_profit=0, reasoning="策略师 API Key 缺失", confidence_score=0)

        # 模拟决策逻辑 (在生产环境中由 AI 生成)
        trend = market_data['technical']['ema_trend']
        rsi = market_data['technical']['rsi']
        
        action = "WAIT"
        trend_zh = "看涨" if trend == "BULLISH" else "看跌"
        if trend == "BULLISH" and rsi < 70: action = "LONG"
        elif trend == "BEARISH" and rsi > 30: action = "SHORT"

        action_zh = "做多" if action == "LONG" else "做空" if action == "SHORT" else "观望"

        return StrategyProposal(
            symbol=market_data['symbol'],
            action=action,
            entry_price=market_data['price'],
            stop_loss=market_data['price'] * (0.99 if action == "LONG" else 1.01),
            take_profit=market_data['price'] * (1.03 if action == "LONG" else 0.97),
            reasoning=f"策略师识别到 {trend_zh} 结构，RSI 位于 {rsi:.1f}。建议 {action_zh}。",
            confidence_score=75 if action != "WAIT" else 0
        )

    async def _call_model_b(self, proposal: StrategyProposal, market_data: dict) -> AuditReport:
        if proposal.action == "WAIT":
            return AuditReport(approved=False, risk_flags=[], liquidity_check="不适用", revised_confidence=0, auditor_comment="策略师未提供可执行方案，保持观望。")
        
        if not self.client_b:
             return AuditReport(approved=True, risk_flags=["无审计员"], liquidity_check="跳过", revised_confidence=proposal.confidence_score, auditor_comment="审计员离线。已绕过审计逻辑，请谨慎。")

        # 审计员风控逻辑
        is_extreme_rsi = market_data['technical']['rsi'] > 80 or market_data['technical']['rsi'] < 20
        approved = not is_extreme_rsi
        
        return AuditReport(
            approved=approved,
            risk_flags=["RSI_极值风险"] if is_extreme_rsi else [],
            liquidity_check="通过",
            revised_confidence=proposal.confidence_score - (10 if is_extreme_rsi else 0),
            auditor_comment="审计员通过趋势对齐检查。" if approved else "审计员驳回：市场处于极端超买/超卖衰竭区。"
        )

    def _adjudicate(self, proposal: StrategyProposal, audit: AuditReport) -> TacticalPlan:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")

        if not audit.approved or proposal.action == "WAIT":
            return TacticalPlan(
                timestamp=ts,
                should_trade=False, 
                symbol=proposal.symbol, 
                action=proposal.action, 
                entry=proposal.entry_price, 
                stop_loss=proposal.stop_loss, 
                take_profit=proposal.take_profit, 
                expected_value=0, 
                rationale=audit.auditor_comment,
                proposal=proposal,
                audit=audit
            )

        # 基础 EV 计算
        p_win = min(max(audit.revised_confidence / 100.0, 0.3), 0.7)
        reward = abs(proposal.take_profit - proposal.entry_price)
        risk = abs(proposal.entry_price - proposal.stop_loss)
        
        rr_ratio = reward / risk if risk > 0 else 0
        ev = (p_win * rr_ratio) - ((1 - p_win) * 1.0)
        
        should_trade = ev > self.config.MIN_EXPECTED_VALUE
        
        status_zh = "执行交易" if should_trade else "拒绝交易 (低期望值)"
        return TacticalPlan(
            timestamp=ts,
            should_trade=should_trade,
            symbol=proposal.symbol,
            action=proposal.action,
            entry=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            expected_value=round(ev, 2),
            rationale=f"最终裁决: {status_zh}。期望值 (EV) 为 {ev:.2f}。",
            proposal=proposal,
            audit=audit
        )

    async def generate_tactics(self, market_data: dict) -> TacticalPlan:
        proposal = await self._call_model_a(market_data)
        audit = await self._call_model_b(proposal, market_data)
        plan = self._adjudicate(proposal, audit)
        return plan
