"""
Utility script to generate a consolidated report file from TradingAgents analysis.
This script extracts all reports and saves them as a single markdown file.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Determine which LLM provider to use (default: deepseek, can be overridden with GOOGLE_API_KEY)
use_google = os.getenv("GOOGLE_API_KEY") is not None
use_deepseek = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")

if use_google:
    # Verify Google API key is loaded
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment. Please set it in .env file or export it.")
    print(f"✅ Google API key loaded (ends with: ...{api_key[-10:]})")
    print(f"   Using provider: Google Gemini")
elif use_deepseek:
    # Verify DeepSeek API key is loaded (DeepSeek uses DEEPSEEK_API_KEY or OPENAI_API_KEY)
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY or OPENAI_API_KEY not found in environment. Please set it in .env file or export it.")
    print(f"✅ API key loaded (ends with: ...{api_key[-10:]})")
    print(f"   Using provider: DeepSeek")
    # Set the API key for DeepSeek (if using DEEPSEEK_API_KEY, also set OPENAI_API_KEY for compatibility)
    if os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
else:
    raise ValueError("No API key found. Please set GOOGLE_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY in environment.")


def generate_consolidated_report(ticker: str, trade_date: str, output_format: str = "markdown"):
    """
    Generate a consolidated report from TradingAgents analysis.
    
    Args:
        ticker: Stock ticker symbol (e.g., "SPY")
        trade_date: Trading date in YYYY-MM-DD format
        output_format: "markdown" or "txt" (default: "markdown")
    """
    # Initialize TradingAgents with Google Gemini or DeepSeek
    config = DEFAULT_CONFIG.copy()
    
    # Determine provider based on available API keys
    use_google = os.getenv("GOOGLE_API_KEY") is not None
    
    if use_google:
        # Configure for Google Gemini
        config["llm_provider"] = "google"  # Use Google Gemini
        config["deep_think_llm"] = "gemini-2.5-pro-preview-06-05"  # Gemini Pro for deep thinking
        config["quick_think_llm"] = "gemini-2.0-flash"  # Gemini Flash for quick thinking
        config["backend_url"] = "https://generativelanguage.googleapis.com/v1"  # Google endpoint
        print("📊 Using Google Gemini models:")
        print(f"   Deep Thinking: {config['deep_think_llm']}")
        print(f"   Quick Thinking: {config['quick_think_llm']}")
    else:
        # Configure for DeepSeek (default)
        config["llm_provider"] = "deepseek"  # Use DeepSeek
        config["deep_think_llm"] = "deepseek-reasoner"  # DeepSeek reasoning model
        config["quick_think_llm"] = "deepseek-chat"  # DeepSeek chat model
        config["backend_url"] = "https://api.deepseek.com/v1"  # DeepSeek endpoint
        print("📊 Using DeepSeek models:")
        print(f"   Deep Thinking: {config['deep_think_llm']}")
        print(f"   Quick Thinking: {config['quick_think_llm']}")
    
    config["max_debate_rounds"] = 1
    config["max_risk_discuss_rounds"] = 1
    
    # Configure data vendors - use Yahoo Finance for company news, Google for global news
    config["data_vendors"]["news_data"] = "yfinance"  # Use Yahoo Finance for company news
    config["tool_vendors"]["get_global_news"] = "google"  # Use Google News for global/macro news (yfinance doesn't support this)
    
    ta = TradingAgentsGraph(
        selected_analysts=["market", "social", "news", "fundamentals"],  # All 4 analysts
        debug=False,  # Set to False to avoid verbose output
        config=config
    )
    
    # Run analysis
    print(f"Running analysis for {ticker} on {trade_date}...")
    final_state, decision = ta.propagate(ticker, trade_date)
    
    # Create output directory
    output_dir = Path("reports") / ticker / trade_date
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate consolidated report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append(f"TRADING AGENTS ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Ticker: {ticker}")
    report_lines.append(f"Analysis Date: {trade_date}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Final Decision
    report_lines.append("# FINAL TRADING DECISION")
    report_lines.append("")
    report_lines.append(f"**Decision: {decision}**")
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")
    
    # Analyst Team Reports
    report_lines.append("# I. ANALYST TEAM REPORTS")
    report_lines.append("")
    
    if final_state.get("market_report"):
        report_lines.append("## Market Analysis")
        report_lines.append("")
        report_lines.append(final_state["market_report"])
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    if final_state.get("sentiment_report"):
        report_lines.append("## Social Sentiment Analysis")
        report_lines.append("")
        report_lines.append(final_state["sentiment_report"])
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    if final_state.get("news_report"):
        report_lines.append("## News Analysis")
        report_lines.append("")
        report_lines.append(final_state["news_report"])
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    if final_state.get("fundamentals_report"):
        report_lines.append("## Fundamentals Analysis")
        report_lines.append("")
        report_lines.append(final_state["fundamentals_report"])
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Research Team Decision
    if final_state.get("investment_debate_state"):
        debate_state = final_state["investment_debate_state"]
        report_lines.append("# II. RESEARCH TEAM DECISION")
        report_lines.append("")
        
        if debate_state.get("bull_history"):
            report_lines.append("## Bull Researcher Analysis")
            report_lines.append("")
            report_lines.append(debate_state["bull_history"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        if debate_state.get("bear_history"):
            report_lines.append("## Bear Researcher Analysis")
            report_lines.append("")
            report_lines.append(debate_state["bear_history"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        if debate_state.get("judge_decision"):
            report_lines.append("## Research Manager Decision")
            report_lines.append("")
            report_lines.append(debate_state["judge_decision"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
    
    # Trading Team
    if final_state.get("trader_investment_plan"):
        report_lines.append("# III. TRADING TEAM PLAN")
        report_lines.append("")
        report_lines.append(final_state["trader_investment_plan"])
        report_lines.append("")
        report_lines.append("---")
        report_lines.append("")
    
    # Risk Management
    if final_state.get("risk_debate_state"):
        risk_state = final_state["risk_debate_state"]
        report_lines.append("# IV. RISK MANAGEMENT TEAM")
        report_lines.append("")
        
        if risk_state.get("risky_history"):
            report_lines.append("## Aggressive Analyst")
            report_lines.append("")
            report_lines.append(risk_state["risky_history"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        if risk_state.get("safe_history"):
            report_lines.append("## Conservative Analyst")
            report_lines.append("")
            report_lines.append(risk_state["safe_history"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        if risk_state.get("neutral_history"):
            report_lines.append("## Neutral Analyst")
            report_lines.append("")
            report_lines.append(risk_state["neutral_history"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
        
        if risk_state.get("judge_decision"):
            report_lines.append("## Portfolio Manager Decision")
            report_lines.append("")
            report_lines.append(risk_state["judge_decision"])
            report_lines.append("")
            report_lines.append("---")
            report_lines.append("")
    
    # Final Trade Decision
    if final_state.get("final_trade_decision"):
        report_lines.append("# V. FINAL TRADE DECISION")
        report_lines.append("")
        report_lines.append(final_state["final_trade_decision"])
        report_lines.append("")
    
    # Save report
    report_content = "\n".join(report_lines)
    
    if output_format == "markdown":
        output_file = output_dir / f"{ticker}_{trade_date}_full_report.md"
    else:
        output_file = output_dir / f"{ticker}_{trade_date}_full_report.txt"
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(report_content)
    
    print(f"\n✅ Consolidated report saved to: {output_file}")
    print(f"📄 Report length: {len(report_content)} characters")
    
    # Also save as JSON for programmatic access
    json_file = output_dir / f"{ticker}_{trade_date}_state.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(final_state, f, indent=2, default=str)
    
    print(f"📊 State data saved to: {json_file}")
    
    return output_file, json_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python generate_report.py <TICKER> <DATE> [format]")
        print("Example: python generate_report.py SPY 2025-01-15 markdown")
        print("Formats: markdown (default) or txt")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    trade_date = sys.argv[2]
    output_format = sys.argv[3] if len(sys.argv) > 3 else "markdown"
    
    try:
        generate_consolidated_report(ticker, trade_date, output_format)
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        import traceback
        traceback.print_exc()
