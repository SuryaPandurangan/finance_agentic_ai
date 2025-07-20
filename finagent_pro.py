# finagent_pro.py

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
from langchain.tools import tool
import yfinance as yf
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

# --- Gemini LLM Setup ---
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "your-service-account.json"
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-preview-05-20", temperature=0.3)

# --- Vector Store Setup ---
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("financial_analysis")

# --- Sentiment Pipeline with Local Model Fallback ---
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


# --- Tool 1: Financial Ratios ---
@tool
def get_financial_ratios(ticker: str) -> str:
    """Fetch key financial ratios like P/E, ROE, EPS for a given stock ticker."""
    stock = yf.Ticker(ticker)
    info = stock.info
    output = f"""
    {ticker.upper()} Key Ratios:
    - P/E: {info.get('trailingPE')}
    - ROE: {info.get('returnOnEquity')}
    - Debt/Equity: {info.get('debtToEquity')}
    - EPS: {info.get('trailingEps')}
    - Revenue (TTM): {info.get('totalRevenue')}
    """
    collection.add(documents=[output], ids=[ticker + "_ratios"])
    return output


# --- Tool 2: Earnings Summary (Placeholder) ---
@tool
def get_earnings_summary(ticker: str) -> str:
    """Return a brief summary of the latest earnings for the given ticker (placeholder for now)."""
    summary = f"Earnings summary for {ticker} from last quarter: Revenue up 12%, Net income flat, Guidance cautious."
    collection.add(documents=[summary], ids=[ticker + "_earnings"])
    return summary


# --- Tool 3: News Sentiment ---
@tool
def analyze_news_sentiment(ticker: str) -> str:
    """Analyze recent news sentiment for the given ticker using a sentiment model."""
    headline = f"{ticker.upper()} under pressure due to rising interest rates."
    score = sentiment_pipeline(headline)
    result = f"Sentiment: {score[0]['label']} ({score[0]['score']:.2f})\nHeadline: {headline}"
    collection.add(documents=[result], ids=[ticker + "_sentiment"])
    return result


# --- Tool 4: Investment Memo Generator ---
@tool
def generate_memo(ticker: str) -> str:
    """Generate an investment memo combining financials, earnings, and sentiment for the given stock."""
    ratios = collection.get(ids=[ticker + "_ratios"])["documents"][0]
    earnings = collection.get(ids=[ticker + "_earnings"])["documents"][0]
    sentiment = collection.get(ids=[ticker + "_sentiment"])["documents"][0]

    prompt = f"""
    Write an investment memo for {ticker}:
    Financials:
    {ratios}

    Earnings Summary:
    {earnings}

    News Sentiment:
    {sentiment}

    Format:
    1. Summary
    2. Key Risks
    3. Recommendation
    """
    return llm.invoke(prompt)


# --- Assemble Tools ---
tools = [
    get_financial_ratios,
    get_earnings_summary,
    analyze_news_sentiment,
    generate_memo,
]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)

# --- Streamlit UI ---
st.set_page_config(page_title="FinAgent Pro", layout="centered")
st.title("📊 FinAgent Pro - AI Financial Analyst")

with st.form("ticker_form"):
    tickers_input = st.text_input(
        "Enter comma-separated ticker symbols (e.g., HDFCBANK, ICICIBANK):"
    )
    submitted = st.form_submit_button("Analyze")

if submitted and tickers_input:
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    for ticker in tickers:
        st.subheader(f"📈 Analysis for {ticker}")
        with st.spinner(f"Analyzing {ticker}..."):
            agent.run(
                f"Get financials, earnings, news sentiment and write investment memo for {ticker}"
            )
            memo = generate_memo.invoke(ticker)
            st.text_area("Investment Memo:", memo, height=300)
