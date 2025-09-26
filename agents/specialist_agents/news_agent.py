"""
News Specialist Agent for Financial Analysis
"""

from typing import Dict, Any, List
from tools.data_sources import YahooFinanceAPI

class NewsSpecialistAgent:
    """Specialist agent for news analysis."""
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze news data for a stock."""
        print(f"News Specialist: Analyzing {symbol}")
        
        try:
            # Get news data
            news_result = self.yahoo_api.get_news(symbol, 10)
            
            if news_result.get("status") != "success":
                return {
                    "specialist": "news",
                    "status": "error",
                    "error": "Failed to fetch news data"
                }
            
            # Analyze news sentiment
            news_data = news_result.get("articles", [])
            sentiment_analysis = self._analyze_sentiment(news_data)
            
            return {
                "specialist": "news",
                "status": "success",
                "analysis": {
                    "sentiment": sentiment_analysis,
                    "articles_analyzed": len(news_data),
                    "key_insights": self._extract_key_insights(news_data)
                }
            }
            
        except Exception as e:
            return {
                "specialist": "news",
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_sentiment(self, news_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sentiment of news articles."""
        if not news_data:
            return {"overall": "neutral", "confidence": 0.0}
        
        positive_keywords = ['growth', 'profit', 'revenue', 'increase', 'positive', 'strong', 'bullish']
        negative_keywords = ['decline', 'loss', 'decrease', 'negative', 'weak', 'bearish']
        
        total_positive = 0
        total_negative = 0
        total_words = 0
        
        for article in news_data:
            text = (article.get("title", "") + " " + article.get("summary", "")).lower()
            words = text.split()
            total_words += len(words)
            
            for word in words:
                if word in positive_keywords:
                    total_positive += 1
                elif word in negative_keywords:
                    total_negative += 1
        
        if total_words == 0:
            return {"overall": "neutral", "confidence": 0.0}
        
        positive_ratio = total_positive / total_words
        negative_ratio = total_negative / total_words
        
        if positive_ratio > negative_ratio:
            sentiment = "positive"
            confidence = positive_ratio
        elif negative_ratio > positive_ratio:
            sentiment = "negative"
            confidence = negative_ratio
        else:
            sentiment = "neutral"
            confidence = 0.0
        
        return {
            "overall": sentiment,
            "confidence": confidence,
            "positive_ratio": positive_ratio,
            "negative_ratio": negative_ratio
        }
    
    def _extract_key_insights(self, news_data: List[Dict[str, Any]]) -> List[str]:
        """Extract key insights from news data."""
        insights = []
        
        if not news_data:
            return insights
        
        # Get most recent articles
        recent_articles = news_data[:3]
        
        for article in recent_articles:
            title = article.get("title", "")
            if title:
                insight = f"Recent news: {title[:80]}..."
                insights.append(insight)
        
        return insights[:3]
