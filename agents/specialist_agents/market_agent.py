"""
Market Specialist Agent for Financial Analysis

Analyzes market trends, price action, and technical indicators.
Uses LLM for intelligent trend analysis that considers context and volume signals,
not just simple price change thresholds. Falls back to rule-based analysis if LLM fails.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tools.data_sources import YahooFinanceAPI
from config import Config

class MarketSpecialistAgent:
    """
    Specialist agent for market data and technical analysis using LLM intelligence.
    
    Analyzes:
    - Price trends (bullish/bearish/neutral)
    - Volume signals (high/moderate/low)
    - Market momentum and technical insights
    
    Uses LLM to provide context-aware trend interpretation that considers
    volume confirmation and market conditions, not just percentage thresholds.
    """
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze market data and technical indicators for a stock."""
        try:
            # Get current price data
            print(f"  Fetching market data for {symbol}...")
            price_result = self.yahoo_api.get_stock_price(symbol)
            
            if price_result.get("status") != "success":
                print(f"  Error: Failed to fetch price data")
                return {
                    "specialist": "market",
                    "status": "error",
                    "error": "Failed to fetch price data"
                }
            
            # Analyze price trends using LLM
            print(f"  Analyzing market trends with LLM...")
            price_analysis = self._analyze_price_trends_with_llm(price_result, symbol)
            
            # Display findings
            print(f"  Current Price: ${price_analysis.get('current_price', 0):.2f}")
            print(f"  Price Change: {price_analysis.get('price_change_percent', 0):.2f}%")
            print(f"  Market Trend: {price_analysis.get('trend', 'N/A')}")
            print(f"  Volume: {price_analysis.get('volume', 0):,}")
            print(f"  Volume Signal: {price_analysis.get('volume_signal', 'N/A')}")
            if price_analysis.get('llm_insights'):
                llm_insights = price_analysis.get('llm_insights', '')
                print(f"  LLM Insights:")
                # Print full insights with proper indentation
                for line in llm_insights.split('\n'):
                    print(f"    {line}")
            
            return {
                "specialist": "market",
                "status": "success",
                "analysis": {
                    "price_trends": price_analysis,
                    "current_price": price_result.get("current_price", 0)
                }
            }
            
        except Exception as e:
            return {
                "specialist": "market",
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_price_trends_with_llm(self, price_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Use LLM to intelligently analyze price trends and provide market insights."""
        current_price = price_data.get("current_price", 0)
        price_change = price_data.get("change_percent", 0)
        volume = price_data.get("volume", 0)
        
        # Use LLM for intelligent trend analysis
        try:
            trend_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a technical analysis expert. Analyze stock price trends and market signals.

Given the price data, provide:
1. Market trend assessment (Strong Bullish/Bullish/Neutral/Bearish/Strong Bearish)
2. Volume analysis (High/Moderate/Low volume and what it means)
3. Brief technical insights about the price action

Consider:
- Price change percentage indicates momentum strength
- Volume confirms trend validity
- Context matters for trend interpretation"""),
                ("human", """Analyze market trends for {symbol}:

Current Price: ${current_price}
Price Change: {price_change:.2f}%
Trading Volume: {volume:,}

Provide a concise trend assessment and technical insights.""")
            ])
            
            chain = trend_prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "current_price": current_price,
                "price_change": price_change,
                "volume": volume
            })
            
            llm_response = response.content.strip()
            
            # Extract trend from LLM response
            llm_lower = llm_response.lower()
            if "strong bullish" in llm_lower or "very bullish" in llm_lower:
                trend = "Strong Bullish"
            elif "bullish" in llm_lower:
                trend = "Bullish"
            elif "strong bearish" in llm_lower or "very bearish" in llm_lower:
                trend = "Strong Bearish"
            elif "bearish" in llm_lower:
                trend = "Bearish"
            else:
                trend = "Neutral"
            
            # Extract volume signal from LLM response
            if "high volume" in llm_lower:
                volume_signal = "High volume activity"
            elif "moderate volume" in llm_lower or "average volume" in llm_lower:
                volume_signal = "Moderate volume"
            else:
                volume_signal = "Low volume"
            
            return {
                "current_price": current_price,
                "price_change_percent": price_change,
                "trend": trend,
                "volume": volume,
                "volume_signal": volume_signal,
                "strength": abs(price_change),
                "llm_insights": llm_response
            }
            
        except Exception as e:
            print(f"  LLM analysis error: {e}, using fallback")
            # Fallback to rule-based analysis
            return self._fallback_trend_analysis(current_price, price_change, volume)
    
    def _fallback_trend_analysis(self, current_price: float, price_change: float, volume: int) -> Dict[str, Any]:
        """Fallback rule-based trend analysis if LLM fails."""
        # Trend analysis
        if price_change > 2:
            trend = "Strong Bullish"
        elif price_change > 0:
            trend = "Bullish"
        elif price_change < -2:
            trend = "Strong Bearish"
        elif price_change < 0:
            trend = "Bearish"
        else:
            trend = "Neutral"
        
        # Volume analysis
        if volume > 1000000:
            volume_signal = "High volume activity"
        elif volume > 500000:
            volume_signal = "Moderate volume"
        else:
            volume_signal = "Low volume"
        
        return {
            "current_price": current_price,
            "price_change_percent": price_change,
            "trend": trend,
            "volume": volume,
            "volume_signal": volume_signal,
            "strength": abs(price_change),
            "llm_insights": ""
        }
