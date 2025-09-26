"""
Minimal Data Sources for Financial Analysis
"""

import yfinance as yf
from typing import Dict, Any, List
from datetime import datetime

class YahooFinanceAPI:
    """Minimal Yahoo Finance API integration."""
    
    def get_stock_price(self, symbol: str) -> Dict[str, Any]:
        """Get current stock price and basic data."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if hist.empty:
                return {"error": f"No data found for {symbol}", "status": "error"}
            
            latest = hist.iloc[-1]
            return {
                "symbol": symbol,
                "current_price": float(latest['Close']),
                "change": float(latest['Close'] - latest['Open']),
                "change_percent": float((latest['Close'] - latest['Open']) / latest['Open'] * 100),
                "volume": int(latest['Volume']),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get basic company information."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "company_name": info.get('longName', 'N/A'),
                "sector": info.get('sector', 'N/A'),
                "industry": info.get('industry', 'N/A'),
                "market_cap": info.get('marketCap', 0),
                "pe_ratio": info.get('trailingPE', 'N/A'),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
    
    def get_news(self, symbol: str, max_articles: int = 10) -> Dict[str, Any]:
        """Get recent news for a stock."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            
            processed_news = []
            for article in news[:max_articles]:
                processed_article = {
                    "title": article.get("title", ""),
                    "summary": article.get("summary", ""),
                    "url": article.get("link", ""),
                    "source": article.get("publisher", ""),
                }
                processed_news.append(processed_article)
            
            return {
                "symbol": symbol,
                "articles": processed_news,
                "count": len(processed_news),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}
