"""
Simple script to analyze SPY for 2025 with rate limit handling.
Uses minimal agents to reduce API calls.
"""
import time
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Create config with minimal settings to reduce API calls
config = DEFAULT_CONFIG.copy()
config["deep_think_llm"] = "gpt-4o-mini"
config["quick_think_llm"] = "gpt-4o-mini"
config["max_debate_rounds"] = 1  # Minimal debates
config["max_risk_discuss_rounds"] = 1  # Minimal risk discussions

# Use only essential analysts to reduce API calls
# Options: ["market"], ["market", "news"], ["market", "news", "fundamentals"]
selected_analysts = ["market", "news"]  # Start with just 2 analysts

print("=" * 60)
print("Analyzing SPY for 2025-01-15")
print(f"Using analysts: {selected_analysts}")
print("=" * 60)
print()

# Initialize with minimal analysts
ta = TradingAgentsGraph(
    selected_analysts=selected_analysts,
    debug=True,
    config=config
)

# Try to run with retry logic
max_retries = 3
retry_delay = 30  # seconds

for attempt in range(max_retries):
    try:
        print(f"\nAttempt {attempt + 1}/{max_retries}...")
        _, decision = ta.propagate("SPY", "2025-01-15")
        print("\n" + "=" * 60)
        print(f"Final Decision: {decision}")
        print("=" * 60)
        print(f"\nReports saved in: results/SPY/2025-01-15/")
        break
    except Exception as e:
        error_str = str(e)
        if "rate_limit" in error_str.lower() or "429" in error_str:
            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                print(f"\n⚠️  Rate limit hit. Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"\n❌ Rate limit error after {max_retries} attempts:")
                print(error_str)
                print("\n💡 Suggestions:")
                print("   1. Wait a few minutes and try again")
                print("   2. Reduce selected_analysts to just ['market']")
                print("   3. Check your rate limits: https://platform.openai.com/account/rate-limits")
        else:
            print(f"\n❌ Error: {error_str}")
            break
