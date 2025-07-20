# news_api.py

import requests


def get_news_headlines(company_name: str, api_key: str, max_results: int = 3) -> list:
    """Fetch recent news headlines with URLs for a company from NewsAPI."""
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": company_name,
        "sortBy": "publishedAt",
        "pageSize": max_results,
        "language": "en",
        "apiKey": api_key,
    }

    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        return [
            {"title": article["title"], "url": article["url"]}
            for article in data.get("articles", [])
        ]
    except Exception as e:
        return [{"title": f"Failed to fetch headlines: {e}", "url": ""}]
