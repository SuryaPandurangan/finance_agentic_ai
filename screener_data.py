# screener_data.py

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.screener.in/company/{}/consolidated/"


def fetch_screener_ratios(ticker: str) -> str:
    """Fetch financial ratios like P/E, ROE, ROCE, Debt/Equity, etc. from Screener.in"""
    url = BASE_URL.format(ticker.upper())
    headers = {"User-Agent": "Mozilla/5.0"}
    res = requests.get(url, headers=headers)
    if res.status_code != 200:
        return f"Failed to fetch data for {ticker} from Screener.in"

    soup = BeautifulSoup(res.text, "html.parser")
    ratios = {}

    try:
        ratio_labels = [el.text.strip() for el in soup.select(".company-ratios .name")]
        ratio_values = [el.text.strip() for el in soup.select(".company-ratios .value")]
        ratios.update(dict(zip(ratio_labels, ratio_values)))
    except Exception as e:
        return f"Error parsing Screener data: {e}"

    fields = [
        "P/E",
        "ROCE",
        "ROE",
        "Debt to equity",
        "EPS",
        "Return on capital employed",
    ]
    summary = f"\n{ticker.upper()} Key Ratios (from Screener.in):\n"
    for field in fields:
        summary += f"- {field}: {ratios.get(field, 'N/A')}\n"

    return summary
