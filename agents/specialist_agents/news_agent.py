"""
News Specialist Agent for Financial Analysis

This agent analyzes news sentiment and market developments for stocks.
It can optionally use PromptChainingWorkflow for enhanced LLM-powered analysis,
or fall back to keyword-based sentiment analysis.
"""

from typing import Dict, Any, List
from tools.data_sources import YahooFinanceAPI

class NewsSpecialistAgent:
    """Specialist agent for news analysis."""
    
    def __init__(self, use_prompt_chaining: bool = False):
        self.yahoo_api = YahooFinanceAPI()
        self.use_prompt_chaining = use_prompt_chaining
        
        # Optionally import prompt chaining workflow
        if use_prompt_chaining:
            try:
                from workflows.prompt_chaining import PromptChainingWorkflow
                self.prompt_chaining_workflow = PromptChainingWorkflow()
            except ImportError:
                self.use_prompt_chaining = False
                print("Warning: PromptChainingWorkflow not available, using basic analysis")
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze news data for a stock. Optionally uses prompt chaining for enhanced analysis."""
        try:
            # Use prompt chaining workflow if enabled
            if self.use_prompt_chaining and hasattr(self, 'prompt_chaining_workflow'):
                print("  Using LLM-powered prompt chaining workflow for enhanced analysis")
                result = self.prompt_chaining_workflow.execute_workflow(symbol, max_articles=10)
                
                if result.get("status") == "success":
                    sentiment = result.get("results", {}).get("sentiment_distribution", {})
                    articles_count = result.get("results", {}).get("articles_processed", 0)
                    llm_summary = result.get("results", {}).get("llm_summary", "")
                    
                    print(f"  Articles Analyzed: {articles_count}")
                    if sentiment:
                        print(f"  Sentiment Distribution: {sentiment}")
                    if llm_summary:
                        print(f"  LLM Summary:")
                        # Print full summary with proper indentation
                        for line in llm_summary.split('\n'):
                            print(f"    {line}")
                    
                    return {
                        "specialist": "news",
                        "status": "success",
                        "analysis": {
                            "sentiment": sentiment,
                            "articles_analyzed": articles_count,
                            "key_entities": result.get("results", {}).get("key_entities", {}),
                            "llm_summary": llm_summary,
                            "workflow": "prompt_chaining"
                        }
                    }
                else:
                    # Fallback to basic analysis if prompt chaining fails
                    print("  Prompt chaining failed, falling back to basic analysis")
            
            # Basic analysis (fallback or when prompt chaining disabled)
            print(f"  Fetching news data for {symbol}...")
            news_result = self.yahoo_api.get_news(symbol, 10)
            
            if news_result.get("status") != "success":
                print(f"  Error: Failed to fetch news data")
                return {
                    "specialist": "news",
                    "status": "error",
                    "error": "Failed to fetch news data"
                }
            
            # Analyze news sentiment
            news_data = news_result.get("articles", [])
            sentiment_analysis = self._analyze_sentiment(news_data)
            key_insights = self._extract_key_insights(news_data)
            
            # Display findings
            print(f"  Articles Analyzed: {len(news_data)}")
            print(f"  Overall Sentiment: {sentiment_analysis.get('overall', 'neutral')} (confidence: {sentiment_analysis.get('confidence', 0):.2f})")
            if key_insights:
                print(f"  Key Insights:")
                for insight in key_insights[:2]:
                    print(f"    - {insight}")
            
            return {
                "specialist": "news",
                "status": "success",
                "analysis": {
                    "sentiment": sentiment_analysis,
                    "articles_analyzed": len(news_data),
                    "key_insights": key_insights,
                    "workflow": "basic"
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
