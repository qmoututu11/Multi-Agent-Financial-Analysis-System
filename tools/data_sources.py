"""
Data Sources for Financial Analysis

Provides access to financial data through multiple APIs:
- Yahoo Finance: Real-time prices, company info, news
- SEC EDGAR: Official SEC filings (10-K, 10-Q, 8-K)

WHY WE NEED REAL DATA SOURCES (not just LLM):
1. **Real-time Accuracy**: LLMs have training cutoffs - they don't know current prices, volumes, or recent events
2. **Prevent Hallucination**: LLMs can make up financial numbers. We fetch real data, then use LLM to analyze it
3. **Current Information**: Market prices change every second - APIs provide live data
4. **Official Data**: SEC filings are regulatory documents - must come from official sources
5. **LLM Role**: LLM analyzes and interprets the real data we fetch, not generates fake data

Architecture:
- APIs fetch REAL data → LLM analyzes and interprets → Intelligent insights
"""

import yfinance as yf
from typing import Dict, Any, List, Optional
import requests
from datetime import datetime, timedelta

class YahooFinanceAPI:
    """
    Yahoo Finance API integration for fetching stock data.
    
    Methods:
        - get_stock_price: Get current price, change, and volume
        - get_company_info: Get company fundamentals (name, sector, P/E ratio, market cap)
        - get_news: Get recent news articles for a stock
    """
    
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
    
    def get_historical_prices(self, symbol: str, period: str = "6mo") -> Dict[str, Any]:
        """
        Get historical price data for trend analysis and forecasting.
        
        Args:
            symbol: Stock ticker symbol
            period: Time period (1mo, 3mo, 6mo, 1y, 2y, 5y, max)
        
        Returns:
            Dict with historical price data including:
            - Price history (dates and prices)
            - Statistics (min, max, average, volatility)
            - Trend indicators
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No historical data found for {symbol}", "status": "error"}
            
            # Calculate statistics
            closes = hist['Close'].tolist()
            volumes = hist['Volume'].tolist()
            dates = hist.index.strftime('%Y-%m-%d').tolist()
            
            # Calculate price changes
            price_changes = []
            for i in range(1, len(closes)):
                change_pct = ((closes[i] - closes[i-1]) / closes[i-1]) * 100
                price_changes.append(change_pct)
            
            # Calculate volatility (standard deviation of returns)
            if price_changes:
                import statistics
                volatility = statistics.stdev(price_changes) if len(price_changes) > 1 else 0.0
            else:
                volatility = 0.0
            
            # Determine trend
            if len(closes) >= 2:
                start_price = closes[0]
                end_price = closes[-1]
                total_change_pct = ((end_price - start_price) / start_price) * 100
                
                # Calculate moving averages (simple)
                recent_prices = closes[-20:] if len(closes) >= 20 else closes
                avg_recent = sum(recent_prices) / len(recent_prices)
                
                if total_change_pct > 5:
                    trend_direction = "uptrend"
                elif total_change_pct < -5:
                    trend_direction = "downtrend"
                else:
                    trend_direction = "sideways"
            else:
                trend_direction = "insufficient_data"
                total_change_pct = 0.0
                avg_recent = closes[0] if closes else 0
            
            return {
                "symbol": symbol,
                "period": period,
                "status": "success",
                "data_points": len(closes),
                "current_price": float(closes[-1]) if closes else 0,
                "start_price": float(closes[0]) if closes else 0,
                "min_price": float(min(closes)) if closes else 0,
                "max_price": float(max(closes)) if closes else 0,
                "avg_price": float(sum(closes) / len(closes)) if closes else 0,
                "avg_recent_price": float(avg_recent),
                "total_change_percent": float(total_change_pct),
                "volatility": float(volatility),
                "trend_direction": trend_direction,
                "recent_prices": closes[-10:] if len(closes) >= 10 else closes,  # Last 10 prices
                "dates": dates[-10:] if len(dates) >= 10 else dates,
                "avg_volume": int(sum(volumes) / len(volumes)) if volumes else 0
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
            
            # Check if news is None or empty
            if not news:
                return {
                    "symbol": symbol,
                    "articles": [],
                    "count": 0,
                    "status": "success",
                    "note": "No news articles available for this symbol"
                }
            
            processed_news = []
            for article in news[:max_articles]:
                # Yahoo Finance news structure: article has 'id' and 'content' keys
                # The actual article data is in article['content']
                content = article.get("content", {})
                
                if not content:
                    # Try direct access if content doesn't exist (fallback)
                    content = article
                
                # Extract fields from content
                title = (
                    content.get("title") or 
                    content.get("headline") or 
                    ""
                )
                
                summary = (
                    content.get("summary") or 
                    content.get("description") or 
                    ""
                )
                
                # URL is nested in canonicalUrl or clickThroughUrl
                url_obj = content.get("canonicalUrl") or content.get("clickThroughUrl") or {}
                url = url_obj.get("url", "") if isinstance(url_obj, dict) else str(url_obj) if url_obj else ""
                
                # Provider is nested
                provider_obj = content.get("provider") or {}
                source = provider_obj.get("displayName", "") if isinstance(provider_obj, dict) else str(provider_obj) if provider_obj else ""
                
                # Only add article if it has at least a title or summary
                if title or summary:
                    processed_article = {
                        "title": title,
                        "summary": summary,
                        "url": url,
                        "source": source,
                    }
                    processed_news.append(processed_article)
            
            # If no valid articles found, return empty but with debug info
            if not processed_news:
                # Debug: print first article structure if available
                if news:
                    print(f"Debug: Yahoo Finance returned {len(news)} items, but no valid articles found.")
                    print(f"Debug: First item keys: {list(news[0].keys()) if news else 'N/A'}")
                    print(f"Debug: First item sample: {str(news[0])[:200] if news else 'N/A'}")
                
                return {
                    "symbol": symbol,
                    "articles": [],
                    "count": 0,
                    "status": "success",
                    "note": "News items found but no valid articles (missing title/summary)"
                }
            
            return {
                "symbol": symbol,
                "articles": processed_news,
                "count": len(processed_news),
                "status": "success"
            }
        except Exception as e:
            return {"error": str(e), "status": "error"}


class SECEdgarAPI:
    """
    SEC EDGAR API integration for fetching official SEC filings.
    
    SEC EDGAR provides free access to company filings including:
    - 10-K: Annual reports (comprehensive financial data)
    - 10-Q: Quarterly reports (quarterly financial data)
    - 8-K: Current reports (material events)
    
    This is official regulatory data - critical for accurate financial analysis.
    """
    
    BASE_URL = "https://data.sec.gov"
    USER_AGENT = "Financial Analysis System contact@example.com"  # SEC requires user agent
    
    def __init__(self):
        """Initialize SEC EDGAR API client."""
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.USER_AGENT})
    
    def get_company_cik(self, symbol: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) for a stock symbol.
        CIK is required to fetch SEC filings.
        """
        try:
            # SEC company tickers mapping (simplified - in production, use official mapping)
            # For now, we'll use a direct lookup approach
            url = f"{self.BASE_URL}/files/company_tickers.json"
            response = self.session.get(url, timeout=10)
            
            if response.status_code == 200:
                companies = response.json()
                # Find company by ticker symbol
                for company in companies.values():
                    if company.get("ticker", "").upper() == symbol.upper():
                        return str(company.get("cik_str", "")).zfill(10)  # Pad CIK to 10 digits
            return None
        except Exception as e:
            print(f"Error fetching CIK for {symbol}: {e}")
            return None
    
    def get_recent_filings(self, symbol: str, filing_type: str = "10-K", max_filings: int = 5) -> Dict[str, Any]:
        """
        Get recent SEC filings for a company.
        
        Args:
            symbol: Stock ticker symbol
            filing_type: Type of filing (10-K, 10-Q, 8-K, etc.)
            max_filings: Maximum number of filings to return
        
        Returns:
            Dict with filings data and status
        """
        try:
            cik = self.get_company_cik(symbol)
            if not cik:
                return {
                    "status": "error",
                    "error": f"Could not find CIK for symbol {symbol}. SEC filings may not be available.",
                    "symbol": symbol
                }
            
            # Get company submissions
            url = f"{self.BASE_URL}/submissions/CIK{cik}.json"
            response = self.session.get(url, timeout=10)
            
            if response.status_code != 200:
                return {
                    "status": "error",
                    "error": f"SEC API error: {response.status_code}",
                    "symbol": symbol
                }
            
            data = response.json()
            filings = data.get("filings", {}).get("recent", {})
            
            # Filter by filing type
            filing_types = filings.get("form", [])
            filing_dates = filings.get("filingDate", [])
            filing_urls = filings.get("primaryDocument", [])
            accession_numbers = filings.get("accessionNumber", [])
            
            recent_filings = []
            for i, form in enumerate(filing_types):
                if form == filing_type and len(recent_filings) < max_filings:
                    accession = accession_numbers[i] if i < len(accession_numbers) else ""
                    filing_url = filing_urls[i] if i < len(filing_urls) else ""
                    
                    # Build full URL
                    if accession and filing_url:
                        url_path = f"{self.BASE_URL}/Archives/edgar/data/{cik}/{accession.replace('-', '')}/{filing_url}"
                    else:
                        url_path = "N/A"
                    
                    recent_filings.append({
                        "filing_type": form,
                        "filing_date": filing_dates[i] if i < len(filing_dates) else "N/A",
                        "accession_number": accession,
                        "url": url_path
                    })
            
            return {
                "status": "success",
                "symbol": symbol,
                "cik": cik,
                "company_name": data.get("name", "N/A"),
                "filing_type": filing_type,
                "filings": recent_filings,
                "count": len(recent_filings)
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol
            }
    
    def get_filing_summary(self, symbol: str, filing_type: str = "10-K") -> Dict[str, Any]:
        """
        Get summary of most recent SEC filing.
        
        This provides high-level information about the filing without downloading
        the full document (which can be very large).
        """
        try:
            filings_result = self.get_recent_filings(symbol, filing_type, max_filings=1)
            
            if filings_result.get("status") != "success" or not filings_result.get("filings"):
                return {
                    "status": "error",
                    "error": f"No {filing_type} filings found for {symbol}",
                    "symbol": symbol
                }
            
            latest_filing = filings_result["filings"][0]
            
            return {
                "status": "success",
                "symbol": symbol,
                "company_name": filings_result.get("company_name", "N/A"),
                "filing_type": filing_type,
                "filing_date": latest_filing.get("filing_date", "N/A"),
                "filing_url": latest_filing.get("url", "N/A"),
                "summary": f"Most recent {filing_type} filing dated {latest_filing.get('filing_date', 'N/A')}",
                "note": "Full filing content would require parsing the HTML/XML document. This provides metadata."
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "symbol": symbol
            }
