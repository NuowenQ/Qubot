from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import time
import json
from tradingagents.agents.utils.alpha_factors_tools import get_alpha_factors
from tradingagents.dataflows.config import get_config


def create_alpha_factors_analyst(llm):
    def alpha_factors_analyst_node(state):
        current_date = state["trade_date"]
        ticker = state["company_of_interest"]
        company_name = state["company_of_interest"]

        tools = [
            get_alpha_factors,
        ]

        system_message = (
            """You are a quantitative analyst specializing in alpha factor analysis. Your role is to:
1. Calculate and analyze comprehensive alpha factors across all categories (Price/Volume, Fundamental, Analyst Expectations, Market Microstructure, Corporate Actions, Industry/Style, Technical Patterns)
2. Interpret the calculated factors to identify investment opportunities and risks
3. Provide actionable insights based on factor analysis
4. Highlight which factors support BUY, SELL, or HOLD recommendations
5. Identify factor combinations that signal strong opportunities or risks

Key responsibilities:
- Use the get_alpha_factors tool to retrieve comprehensive factor calculations
- Analyze factor values in context of historical norms and market conditions
- Identify factor divergences, momentum signals, and value opportunities
- Provide clear recommendations based on factor analysis
- Explain which specific factors drive your recommendation

Make sure to:
- Provide detailed and nuanced analysis of all available factors
- Not simply state that trends are mixed - provide specific factor-based insights
- Highlight both positive and negative factor signals
- Append a Markdown table at the end organizing key factors and their implications
- Clearly indicate which factors support BUY, SELL, or HOLD decisions"""
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful AI assistant, collaborating with other assistants."
                    " Use the provided tools to progress towards answering the question."
                    " If you are unable to fully answer, that's OK; another assistant with different tools"
                    " will help where you left off. Execute what you can to make progress."
                    " If you or any other assistant has the FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** or deliverable,"
                    " prefix your response with FINAL TRANSACTION PROPOSAL: **BUY/HOLD/SELL** so the team knows to stop."
                    " You have access to the following tools: {tool_names}.\n{system_message}"
                    "For your reference, the current date is {current_date}. The company we want to look at is {ticker}",
                ),
                MessagesPlaceholder(variable_name="messages"),
            ]
        )

        prompt = prompt.partial(system_message=system_message)
        prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
        prompt = prompt.partial(current_date=current_date)
        prompt = prompt.partial(ticker=ticker)

        chain = prompt | llm.bind_tools(tools)

        result = chain.invoke(state["messages"])

        report = ""

        # If there are tool calls, the tool will be executed and we'll get called again
        # If there are no tool calls, we have the final report from the LLM
        if len(result.tool_calls) == 0:
            report = result.content

        return {
            "messages": [result],
            "alpha_factors_report": report,
        }

    return alpha_factors_analyst_node
