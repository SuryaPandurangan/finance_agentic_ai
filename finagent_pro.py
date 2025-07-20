# finagent_pro.py (updated)

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import os
import chromadb
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sector_template import get_sector_specific_prompt
from screener_data import fetch_screener_ratios
from news_api import get_news_headlines
from dotenv import load_dotenv

load_dotenv()

# --- Gemini LLM Setup ---
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-service-account.json"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.3)

# --- Vector Store Setup ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("financial_analysis")

# --- Sentiment Pipeline ---
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
local_path = f"./models/{model_name.replace('/', '_')}"

if os.path.exists(local_path):
    tokenizer = AutoTokenizer.from_pretrained(local_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_path)
else:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    os.makedirs(local_path, exist_ok=True)
    tokenizer.save_pretrained(local_path)
    model.save_pretrained(local_path)

sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)


# --- Tool 1: Financial Ratios (Screener) ---
@tool
def get_financial_ratios(ticker: str) -> str:
    """Fetch key financial ratios from Screener.in for a given Indian stock ticker."""
    output = fetch_screener_ratios(ticker)
    collection.add(documents=[output], ids=[ticker + "_ratios"])
    return output


# --- Tool 2: News Sentiment with Real Headlines ---
@tool
def analyze_news_sentiment(ticker: str) -> str:
    """Analyze real news sentiment for a given company using NewsAPI."""
    headlines = get_news_headlines(ticker, os.getenv("NEWS_API_KEY"))
    results = []
    total_score = 0

    for item in headlines:
        score = sentiment_pipeline(item["title"])
        sentiment_text = f"Sentiment: {score[0]['label']} ({score[0]['score']:.2f})"
        total_score += (
            score[0]["score"]
            if score[0]["label"].upper() == "POSITIVE"
            else -score[0]["score"]
        )
        link = f"[Read more]({item['url']})" if item["url"] else ""
        results.append(f"Headline: {item['title']}\n{sentiment_text}\n{link}\n")

    avg_score = total_score / len(headlines) if headlines else 0
    summary = f"**Confidence Score (Sentiment Net): {avg_score:.2f}**\n\n" + "\n".join(
        results
    )
    collection.add(documents=[summary], ids=[ticker + "_sentiment"])
    return summary


# --- Tool 3: Investment Memo Generator ---
@tool
def generate_memo(ticker: str) -> str:
    """Generate a sector-specific investment memo combining financials and news sentiment."""
    ratios = collection.get(ids=[ticker + "_ratios"])["documents"][0]
    sentiment = collection.get(ids=[ticker + "_sentiment"])["documents"][0]
    sector = "general"

    missing = []
    for key in ["P/E", "EPS", "Debt to equity"]:
        if key not in ratios:
            missing.append(key)

    highlight_missing = "**Missing Data:** " + (
        ", ".join(missing) if missing else "None"
    )
    highlight_available = "**Available Data:** ROE, ROCE"

    preamble = f"{highlight_available}  \n{highlight_missing}\n\n"
    prompt = preamble + get_sector_specific_prompt(
        ticker, ratios, "(Earnings skipped)", sentiment, sector
    )
    return llm.invoke(prompt)


# --- Assemble Agent ---
tools = [get_financial_ratios, analyze_news_sentiment, generate_memo]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="FinAgent Pro", layout="centered")
st.title("📊 FinAgent Pro - AI Financial Analyst (India Edition)")

with st.form("ticker_form"):
    tickers_input = st.text_input(
        "Enter comma-separated NSE stock symbols (e.g., HDFCBANK, RELIANCE):"
    )
    submitted = st.form_submit_button("Analyze")

if submitted and tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    for ticker in tickers:
        st.subheader(f"📈 Analysis for {ticker}")
        with st.spinner(f"Analyzing {ticker}..."):
            agent.run(
                f"Get financials, news sentiment and write investment memo for {ticker}"
            )
            memo = generate_memo.invoke(ticker)
            memo_text = memo.content if hasattr(memo, "content") else str(memo)
            st.markdown(memo_text, unsafe_allow_html=True)
