# 📊 FinAgent Pro – AI-Powered Investment Memo Generator

**FinAgent Pro** is an intelligent financial analyst assistant that uses **LangChain**, **Gemini LLM**, and real-time data from **Screener.in** and **NewsAPI** to generate sector-aware investment memos for Indian listed companies. Built with Python, Streamlit, ChromaDB, and HuggingFace.

---

## 🚀 Features

* 🔍 **Company-specific Analysis** – Just enter a stock symbol (e.g., `HDFCBANK`, `ICICIBANK`)
* 📈 **Key Ratio Extraction** – Pulls financials like ROE, ROCE, P/E, EPS, etc. from Screener.in
* 📰 **News Sentiment Integration** – Uses NewsAPI + HuggingFace DistilBERT to score headlines
* 🧠 **Agentic AI Reasoning** – LangChain agent uses tools + context to invoke Gemini for memo writing
* 📄 **Investment Memo Output** – Markdown-rendered memo with:

  * Summary, Key Risks, Recommendation
  * Missing data indicators
  * Confidence score from sentiment analysis
  * Clickable news links

---

## 🛠️ Tech Stack

| Component       | Tech Used                                                      |
| --------------- | -------------------------------------------------------------- |
| LLM             | Gemini Pro (via LangChain + Google GenAI)                      |
| Agent Framework | LangChain agent w/ custom tools                                |
| Data Sources    | Screener.in (financials), NewsAPI (headlines)                  |
| Vector Store    | ChromaDB                                                       |
| Sentiment Model | Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` |
| UI              | Streamlit                                                      |

---

## 📁 Folder Structure

```
/finance_agentic_ai/
├── finagent_pro.py         # Main Streamlit app
├── news_api.py             # NewsAPI integration
├── screener_data.py        # Screener.in web scraping
├── sector_template.py      # Sector-specific LLM prompts
├── models/                 # Local sentiment model storage
├── .env / service-account  # Your Gemini/Google credentials
```

---

## 🧑‍💻 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/your-repo/finance_agentic_ai.git
cd finance_agentic_ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure

* Replace `your-service-account.json` with your Gemini credentials
* Add your NewsAPI key to `finagent_pro.py`:

```python
NEWS_API_KEY = "your_actual_key"
```

### 3. Run It

```bash
streamlit run finagent_pro.py
```

---

## 🏁 Example Output

```
📈 Analysis for ICICIBANK

**Available Data:** ROE, ROCE
**Missing Data:** P/E, EPS, Debt to equity
**Confidence Score (Sentiment Net): 0.78**

### Summary
ICICI Bank has an ROE of 18%... [detailed memo continues]

### Key Risks
- Market-wide sentiment drag
- Absence of key leverage metrics

### Recommendation
DEFER – Further financial and sectoral insight needed.
```

---

## 📌 Future Improvements

* PDF export of memo
* Sector auto-detection
* Add charts with `plotly`
* Notion/Airtable sync
* Daily scheduler for tracked tickers

---

## 📃 License

MIT License

---

## 🙌 Credits

* Screener.in (data)
* HuggingFace Transformers
* Google Gemini + LangChain
* NewsAPI.org

---

Made with 💼 for financial professionals and investors.

---
