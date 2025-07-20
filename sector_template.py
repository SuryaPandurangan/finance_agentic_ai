# sector_template.py


def get_sector_specific_prompt(
    ticker: str, ratios: str, earnings: str, sentiment: str, sector: str
) -> str:
    sector = sector.lower()

    if "financial" in sector:
        return f"""
        Generate an investment memo for the bank {ticker}:
        Focus areas:
        - NPA trends, CASA ratio, Net Interest Margin
        - YoY loan book growth, deposit base, and regulatory risks

        Financials:
        {ratios}

        Earnings Summary:
        {earnings}

        News Sentiment:
        {sentiment}

        Format:
        1. Executive Summary
        2. Key Risks
        3. Investment Recommendation
        """

    elif "healthcare" in sector or "pharma" in sector:
        return f"""
        Generate an investment memo for the pharma company {ticker}:
        Focus areas:
        - R&D expenditure, product pipeline approvals, FDA risk
        - Profit margins, pricing regulations, global expansion

        Financials:
        {ratios}

        Earnings Summary:
        {earnings}

        News Sentiment:
        {sentiment}

        Format:
        1. Executive Summary
        2. Risk Factors
        3. Investment Recommendation
        """

    else:
        return f"""
        Generate a general investment memo for {ticker} using the data below.

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
