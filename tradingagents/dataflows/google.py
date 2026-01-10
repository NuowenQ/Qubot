from typing import Annotated
from datetime import datetime
from dateutil.relativedelta import relativedelta
from .googlenews_utils import getNewsData


def get_google_news(
    query: Annotated[str, "Query to search with"],
    curr_date: Annotated[str, "Curr date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "how many days to look back"],
) -> str:
    query = query.replace(" ", "+")

    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")

    news_results = getNewsData(query, before, curr_date)

    news_str = ""

    for news in news_results:
        news_str += (
            f"### {news['title']} (source: {news['source']}) \n\n{news['snippet']}\n\n"
        )

    if len(news_results) == 0:
        return ""

    return f"## {query} Google News, from {before} to {curr_date}:\n\n{news_str}"


def get_global_news_google(
    curr_date: Annotated[str, "Current date in yyyy-mm-dd format"],
    look_back_days: Annotated[int, "Number of days to look back"] = 7,
    limit: Annotated[int, "Maximum number of articles to return"] = 5,
) -> str:
    """
    Get global/macroeconomics news from Google News.
    This is a wrapper for Google News that searches for general market and economic news.
    
    Args:
        curr_date: Current date in yyyy-mm-dd format
        look_back_days: Number of days to look back (default 7)
        limit: Maximum number of articles to return (default 5)
    
    Returns:
        Formatted string containing global news articles
    """
    # Use generic queries for global/macro news
    queries = [
        "market news economy",
        "macroeconomic news",
        "financial markets",
        "global economy",
        "stock market"
    ]
    
    start_date = datetime.strptime(curr_date, "%Y-%m-%d")
    before = start_date - relativedelta(days=look_back_days)
    before = before.strftime("%Y-%m-%d")
    
    all_news = []
    seen_titles = set()  # Avoid duplicates
    
    # Search multiple queries and combine results
    for query in queries:
        try:
            news_results = getNewsData(query, before, curr_date)
            for news in news_results:
                title = news.get('title', '')
                if title and title not in seen_titles:
                    all_news.append(news)
                    seen_titles.add(title)
                    if len(all_news) >= limit:
                        break
            if len(all_news) >= limit:
                break
        except Exception as e:
            # Continue with other queries if one fails
            continue
    
    if len(all_news) == 0:
        return f"No global news found between {before} and {curr_date}"
    
    # Limit results
    all_news = all_news[:limit]
    
    news_str = ""
    for news in all_news:
        news_str += (
            f"### {news['title']} (source: {news.get('source', 'Unknown')})\n\n"
            f"{news.get('snippet', news.get('description', ''))}\n\n"
        )
    
    return f"## Global Market News from Google, from {before} to {curr_date}:\n\n{news_str}"