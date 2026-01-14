# TradingAgents/graph/signal_processing.py

from langchain_openai import ChatOpenAI


class SignalProcessor:
    """Processes trading signals to extract actionable decisions."""

    def __init__(self, quick_thinking_llm: ChatOpenAI):
        """Initialize with an LLM for processing."""
        self.quick_thinking_llm = quick_thinking_llm

    def process_signal(self, full_signal: str) -> str:
        """
        Process a full trading signal to extract the core decision.

        Args:
            full_signal: Complete trading signal text

        Returns:
            Extracted decision (STRONG_BUY, BUY, HOLD_LONG, HOLD_CASH, REDUCE, SELL, or SHORT)
        """
        messages = [
            (
                "system",
                """You are an efficient assistant designed to analyze paragraphs or financial reports provided by a group of analysts.
Your task is to extract the investment decision from one of these 7 actions:

1. STRONG_BUY - High conviction bullish entry, strong upside expected
2. BUY - Standard bullish entry, favorable risk/reward
3. HOLD_LONG - Maintain current long position, no action needed
4. HOLD_CASH - Stay in cash, wait for better opportunity
5. REDUCE - Partial exit, take some profits or reduce exposure
6. SELL - Full exit of long position, avoid downside
7. SHORT - Enter bearish position, expect price decline

IMPORTANT DISTINCTIONS:
- HOLD_LONG means you own the stock and should keep it
- HOLD_CASH means you don't own it and should wait
- REDUCE means trim position size, not full exit
- SELL means complete exit
- SHORT means actively bet against the stock

Provide ONLY the extracted decision (one of the 7 actions above) as your output, without adding any additional text or information.""",
            ),
            ("human", full_signal),
        ]

        return self.quick_thinking_llm.invoke(messages).content
