import time
import json


def create_research_manager(llm, memory):
    def research_manager_node(state) -> dict:
        history = state["investment_debate_state"].get("history", "")
        market_research_report = state["market_report"]
        sentiment_report = state["sentiment_report"]
        news_report = state["news_report"]
        fundamentals_report = state["fundamentals_report"]

        investment_debate_state = state["investment_debate_state"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the portfolio manager and debate facilitator, your role is to critically evaluate this round of debate and make a definitive decision from these 7 actions:

1. STRONG_BUY - High conviction bullish entry
2. BUY - Standard bullish entry
3. HOLD_LONG - Maintain current position
4. HOLD_CASH - Wait in cash for better opportunity
5. REDUCE - Partial exit to manage risk
6. SELL - Full exit of position
7. SHORT - Enter bearish position

Summarize the key points from both sides concisely, focusing on the most compelling evidence or reasoning. Your recommendation must be clear and actionable.

CRITICAL GUIDELINES:
- **Don't default to HOLD_CASH or HOLD_LONG as a compromise** - choose the action with highest expected value
- If the bull case has merit, specify BUY or STRONG_BUY (don't say "wait for pullback" - that's HOLD_CASH)
- If the bear case wins, specify SELL, REDUCE, or SHORT (be explicit about the action)
- HOLD_CASH means "actively waiting for a better entry" - it requires justification that waiting has positive EV vs entering now
- Consider: What is the opportunity cost of waiting? Could the stock run away?

Additionally, develop a detailed investment plan for the trader. This should include:

Your Recommendation: One of the 7 actions above, supported by the most convincing arguments
Rationale: Explain why these arguments lead to your conclusion, including expected value reasoning
Strategic Actions: Concrete steps for implementing the recommendation
Take into account your past mistakes on similar situations. Use these insights to refine your decision-making and ensure you are learning and improving. Pay attention if your past errors show excessive caution or missed opportunities.

Present your analysis conversationally, as if speaking naturally, without special formatting.

Here are your past reflections on mistakes:
\"{past_memory_str}\"

Here is the debate:
Debate History:
{history}"""
        response = llm.invoke(prompt)

        new_investment_debate_state = {
            "judge_decision": response.content,
            "history": investment_debate_state.get("history", ""),
            "bear_history": investment_debate_state.get("bear_history", ""),
            "bull_history": investment_debate_state.get("bull_history", ""),
            "current_response": response.content,
            "count": investment_debate_state["count"],
        }

        return {
            "investment_debate_state": new_investment_debate_state,
            "investment_plan": response.content,
        }

    return research_manager_node
