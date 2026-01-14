import time
import json


def create_risk_manager(llm, memory):
    def risk_manager_node(state) -> dict:

        company_name = state["company_of_interest"]

        history = state["risk_debate_state"]["history"]
        risk_debate_state = state["risk_debate_state"]
        market_research_report = state["market_report"]
        news_report = state["news_report"]
        fundamentals_report = state["news_report"]
        sentiment_report = state["sentiment_report"]
        trader_plan = state["investment_plan"]

        curr_situation = f"{market_research_report}\n\n{sentiment_report}\n\n{news_report}\n\n{fundamentals_report}"
        past_memories = memory.get_memories(curr_situation, n_matches=2)

        past_memory_str = ""
        for i, rec in enumerate(past_memories, 1):
            past_memory_str += rec["recommendation"] + "\n\n"

        prompt = f"""As the Risk Management Judge and Debate Facilitator, your goal is to evaluate the debate between three risk analysts—Risky, Neutral, and Safe/Conservative—and determine the best course of action for the trader.

Your decision must be one of these 7 actions:
1. STRONG_BUY - High conviction bullish entry
2. BUY - Standard bullish entry
3. HOLD_LONG - Maintain current position
4. HOLD_CASH - Wait in cash for better opportunity
5. REDUCE - Partial exit to manage risk
6. SELL - Full exit of position
7. SHORT - Enter bearish position

CRITICAL: You must think probabilistically, not just narratively. For each potential action, estimate:
- Upside probability and magnitude (% gain if correct)
- Downside probability and magnitude (% loss if wrong)
- Expected value (EV) = (Upside% × UpsideGain) - (Downside% × DownsideLoss)

Guidelines for Decision-Making:
1. **Calculate Expected Value**: Explicitly estimate probabilities and expected returns for BUY vs HOLD_CASH vs SELL scenarios
2. **Summarize Key Arguments**: Extract the strongest points from each analyst, focusing on relevance to the context
3. **Provide Rationale**: Support your recommendation with direct quotes, counterarguments, AND quantitative EV reasoning
4. **Avoid Narrative Risk Overweighting**: Don't let vivid downside stories bias you - weigh them by actual probability
5. **Refine the Trader's Plan**: Start with the trader's original plan, **{trader_plan}**, and adjust it based on the analysts' insights
6. **Learn from Past Mistakes**: Use lessons from **{past_memory_str}** to address prior misjudgments. Pay special attention if past errors show a pattern of excessive caution or missed opportunities.

IMPORTANT: Default to action (BUY/SELL/REDUCE) when EV is positive. Choose HOLD_CASH only when waiting has higher EV than entering. Don't confuse prudence with inaction.

Deliverables:
- EV calculation for top 2-3 action candidates (show your math)
- A clear and actionable recommendation from the 7 actions above
- Detailed reasoning anchored in the debate, EV analysis, and past reflections

---

**Analysts Debate History:**  
{history}

---

Focus on actionable insights and continuous improvement. Build on past lessons, critically evaluate all perspectives, and ensure each decision advances better outcomes."""

        response = llm.invoke(prompt)

        new_risk_debate_state = {
            "judge_decision": response.content,
            "history": risk_debate_state["history"],
            "risky_history": risk_debate_state["risky_history"],
            "safe_history": risk_debate_state["safe_history"],
            "neutral_history": risk_debate_state["neutral_history"],
            "latest_speaker": "Judge",
            "current_risky_response": risk_debate_state["current_risky_response"],
            "current_safe_response": risk_debate_state["current_safe_response"],
            "current_neutral_response": risk_debate_state["current_neutral_response"],
            "count": risk_debate_state["count"],
        }

        return {
            "risk_debate_state": new_risk_debate_state,
            "final_trade_decision": response.content,
        }

    return risk_manager_node
