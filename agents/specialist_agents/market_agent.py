"""
Market Specialist Agent for Financial Analysis
"""

from typing import Dict, Any
from tools.data_sources import YahooFinanceAPI

class MarketSpecialistAgent:
    """Specialist agent for market data and technical analysis."""
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze market data and technical indicators for a stock."""
        print(f"Market Specialist: Analyzing {symbol}")
        
        try:
            # Get current price data
            price_result = self.yahoo_api.get_stock_price(symbol)
            
            if price_result.get("status") != "success":
                return {
                    "specialist": "market",
                    "status": "error",
                    "error": "Failed to fetch price data"
                }
            
            # Analyze price trends
            price_analysis = self._analyze_price_trends(price_result)
            
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
    
    def _analyze_price_trends(self, price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze price trends."""
        current_price = price_data.get("current_price", 0)
        price_change = price_data.get("change_percent", 0)
        volume = price_data.get("volume", 0)
        
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
        if volume > 1000000:  # Simple threshold
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
            "strength": abs(price_change)
        }
