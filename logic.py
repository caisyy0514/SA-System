import json
import logging
from pydantic import BaseModel
from typing import Optional, Literal
from google import genai 
from openai import AsyncOpenAI
from config import BotConfig

logger = logging.getLogger("sentinel.logic")

# --- Structured Output Definitions ---

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
    risk_flags: list[str]
    liquidity_check: str
    revised_confidence: int # 0-100
    auditor_comment: str

class TacticalPlan(BaseModel):
    should_trade: bool
    symbol: str
    action: str
    entry: float
    stop_loss: float
    take_profit: float
    expected_value: float
    rationale: str

# --- Debate Logic ---

class DebateManager:
    def __init__(self, config: BotConfig):
        self.config = config
        self.client_a = None
        self.client_b = None

        # Model A: Gemini (Strategist)
        if config.GEMINI_API_KEY:
            self.client_a = genai.Client(api_key=config.GEMINI_API_KEY)
        
        # Model B: DeepSeek (Auditor)
        if config.DEEPSEEK_API_KEY:
            self.client_b = AsyncOpenAI(
                api_key=config.DEEPSEEK_API_KEY, 
                base_url="https://api.deepseek.com/v1"
            )

    async def _call_model_a(self, market_data: dict) -> StrategyProposal:
        if not self.client_a:
             return StrategyProposal(symbol=market_data['symbol'], action="WAIT", entry_price=0, stop_loss=0, take_profit=0, reasoning="API Key Missing", confidence_score=0)

        prompt = f"""
        Role: Senior Crypto Scalper.
        Task: Analyze data and propose a setup.
        Context: OKX Perp {market_data['symbol']}.
        Data: {json.dumps(market_data, indent=2)}
        
        Rules:
        1. Follow the Trend (EMA).
        2. Check Orderbook Imbalance.
        3. ATR used for volatility context.
        
        Output JSON only matching StrategyProposal schema.
        """
        try:
            # Placeholder for actual Gemini Call
            # response = await self.client_a.models.generate_content(...)
            
            # MOCK RETURN for demonstration
            return StrategyProposal(
                symbol=market_data['symbol'],
                action="LONG" if market_data['technical']['ema_trend'] == "BULLISH" else "SHORT",
                entry_price=market_data['price'],
                stop_loss=market_data['price'] * 0.99 if market_data['technical']['ema_trend'] == "BULLISH" else market_data['price'] * 1.01,
                take_profit=market_data['price'] * 1.02 if market_data['technical']['ema_trend'] == "BULLISH" else market_data['price'] * 0.98,
                reasoning="Trend following with EMA alignment",
                confidence_score=75
            )
        except Exception as e:
            logger.error(f"Model A failed: {e}")
            return StrategyProposal(symbol=market_data['symbol'], action="WAIT", entry_price=0, stop_loss=0, take_profit=0, reasoning=f"Error: {str(e)}", confidence_score=0)

    async def _call_model_b(self, proposal: StrategyProposal, market_data: dict) -> AuditReport:
        if proposal.action == "WAIT":
            return AuditReport(approved=False, risk_flags=[], liquidity_check="N/A", revised_confidence=0, auditor_comment="No trade proposed")
        
        if not self.client_b:
            # If no B model, auto-approve with warning
             return AuditReport(approved=True, risk_flags=["NO_AUDITOR"], liquidity_check="Skipped", revised_confidence=proposal.confidence_score, auditor_comment="B Model Key Missing, bypassing audit.")

        prompt = f"""
        Role: Risk Manager & Skeptic.
        Task: Audit this trade proposal. Look for liquidity traps, counter-trend risks.
        Proposal: {proposal.model_dump_json()}
        Market Data: {json.dumps(market_data['technical'], indent=2)}
        
        Output JSON only matching AuditReport schema.
        """
        try:
            # Placeholder for DeepSeek Call
            # response = await self.client_b.chat.completions.create(...)
            
            # MOCK RETURN
            return AuditReport(
                approved=True,
                risk_flags=["High Volatility"],
                liquidity_check="Pass",
                revised_confidence=proposal.confidence_score - 5,
                auditor_comment="Approved but slight deduction for ATR expansion."
            )
        except Exception as e:
            logger.error(f"Model B failed: {e}")
            return AuditReport(approved=False, risk_flags=["System Error"], liquidity_check="Fail", revised_confidence=0, auditor_comment=f"Error: {str(e)}")

    def _adjudicate(self, proposal: StrategyProposal, audit: AuditReport) -> TacticalPlan:
        if not audit.approved or proposal.action == "WAIT":
            return TacticalPlan(should_trade=False, symbol=proposal.symbol, action="WAIT", entry=0, sl=0, tp=0, expected_value=0, rationale="Rejected by auditor")

        p_win = min(max(audit.revised_confidence / 100.0, 0.2), 0.8) 
        p_loss = 1.0 - p_win
        
        reward = abs(proposal.take_profit - proposal.entry_price)
        risk = abs(proposal.entry_price - proposal.stop_loss)
        
        if risk == 0: return TacticalPlan(should_trade=False, symbol=proposal.symbol, action="WAIT", entry=0, sl=0, tp=0, expected_value=0, rationale="Zero risk denominator")

        rr_ratio = reward / risk
        ev = (p_win * rr_ratio) - (p_loss * 1.0)
        
        should_trade = ev > self.config.MIN_EXPECTED_VALUE
        
        return TacticalPlan(
            should_trade=should_trade,
            symbol=proposal.symbol,
            action=proposal.action,
            entry=proposal.entry_price,
            stop_loss=proposal.stop_loss,
            take_profit=proposal.take_profit,
            expected_value=round(ev, 2),
            rationale=f"EV: {ev:.2f} (Win%: {p_win*100}%, RR: {rr_ratio:.2f}). {audit.auditor_comment}"
        )

    async def generate_tactics(self, market_data: dict) -> TacticalPlan:
        proposal = await self._call_model_a(market_data)
        audit = await self._call_model_b(proposal, market_data)
        plan = self._adjudicate(proposal, audit)
        return plan
