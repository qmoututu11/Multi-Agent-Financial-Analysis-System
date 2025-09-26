"""
LangChain Tools for Financial Analysis
"""

from typing import Dict, Any, List
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from .data_sources import YahooFinanceAPI

class StockPriceInput(BaseModel):
    """Input for stock price tool."""
    symbol: str = Field(description="Stock symbol to get price for")

class StockPriceTool(BaseTool):
    """Tool for fetching stock price data."""
    name: str = "get_stock_price"
    description: str = "Get current stock price and basic market data for a given symbol"
    args_schema: type = StockPriceInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._yahoo_api = YahooFinanceAPI()
    
    def _run(self, symbol: str) -> str:
        """Execute the tool."""
        try:
            result = self._yahoo_api.get_stock_price(symbol)
            if result.get("status") == "success":
                return f"Stock: {symbol}\nPrice: ${result['current_price']:.2f}\nChange: {result['change_percent']:+.2f}%\nVolume: {result['volume']:,}"
            else:
                return f"Error fetching price for {symbol}: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error: {str(e)}"

class CompanyInfoInput(BaseModel):
    """Input for company info tool."""
    symbol: str = Field(description="Stock symbol to get company info for")

class CompanyInfoTool(BaseTool):
    """Tool for fetching company information."""
    name: str = "get_company_info"
    description: str = "Get company information including name, sector, and market cap"
    args_schema: type = CompanyInfoInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._yahoo_api = YahooFinanceAPI()
    
    def _run(self, symbol: str) -> str:
        """Execute the tool."""
        try:
            result = self._yahoo_api.get_company_info(symbol)
            if result.get("status") == "success":
                return f"Company: {result['company_name']}\nSector: {result['sector']}\nIndustry: {result['industry']}\nMarket Cap: ${result['market_cap']:,}"
            else:
                return f"Error fetching company info for {symbol}: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error: {str(e)}"

class NewsAnalysisInput(BaseModel):
    """Input for news analysis tool."""
    symbol: str = Field(description="Stock symbol to analyze news for")
    max_articles: int = Field(default=5, description="Maximum number of articles to analyze")

class NewsAnalysisTool(BaseTool):
    """Tool for analyzing news sentiment."""
    name: str = "analyze_news"
    description: str = "Analyze news sentiment for a stock"
    args_schema: type = NewsAnalysisInput
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._yahoo_api = YahooFinanceAPI()
    
    def _run(self, symbol: str, max_articles: int = 5) -> str:
        """Execute the tool."""
        try:
            result = self._yahoo_api.get_news(symbol, max_articles)
            if result.get("status") == "success":
                articles = result.get("articles", [])
                if not articles:
                    return f"No recent news found for {symbol}"
                
                # Simple sentiment analysis
                positive_keywords = ["growth", "profit", "revenue", "increase", "positive", "strong", "beat", "exceed"]
                negative_keywords = ["loss", "decline", "decrease", "negative", "weak", "miss", "disappoint"]
                
                sentiment_scores = []
                for article in articles[:3]:  # Analyze first 3 articles
                    text = (article.get("title", "") + " " + article.get("summary", "")).lower()
                    pos_count = sum(1 for word in positive_keywords if word in text)
                    neg_count = sum(1 for word in negative_keywords if word in text)
                    
                    if pos_count > neg_count:
                        sentiment_scores.append(1)
                    elif neg_count > pos_count:
                        sentiment_scores.append(-1)
                    else:
                        sentiment_scores.append(0)
                
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
                sentiment = "positive" if avg_sentiment > 0.2 else "negative" if avg_sentiment < -0.2 else "neutral"
                
                return f"News Analysis for {symbol}:\nSentiment: {sentiment}\nArticles analyzed: {len(articles)}"
            else:
                return f"Error fetching news for {symbol}: {result.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Error: {str(e)}"

def get_financial_tools() -> List[BaseTool]:
    """Get all financial analysis tools."""
    return [
        StockPriceTool(),
        CompanyInfoTool(),
        NewsAnalysisTool()
    ]
