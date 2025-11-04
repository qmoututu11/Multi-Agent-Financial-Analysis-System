"""
Financial Forecast Specialist Agent

Analyzes historical price trends and generates forward-looking forecasts.
Uses LLM for intelligent trend analysis and prediction based on historical data,
market conditions, and company fundamentals.
"""

from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tools.data_sources import YahooFinanceAPI
from config import Config

class ForecastSpecialistAgent:
    """
    Specialist agent for financial forecasting using LLM intelligence.
    
    Analyzes:
    - Historical price trends and patterns
    - Volatility and risk indicators
    - Forward-looking price predictions
    - Support and resistance levels
    - Forecast confidence intervals
    
    Uses LLM to provide intelligent forecasting based on:
    - Historical price patterns
    - Current market conditions
    - Technical indicators
    - Company fundamentals
    """
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze historical trends and generate financial forecasts."""
        try:
            # Get current price data
            print(f"  Fetching current price data for {symbol}...")
            price_result = self.yahoo_api.get_stock_price(symbol)
            
            # Get historical price data for trend analysis
            print(f"  Fetching historical price data (6 months)...")
            historical_data = self.yahoo_api.get_historical_prices(symbol, period="6mo")
            
            if historical_data.get("status") != "success":
                print(f"  Error: Failed to fetch historical data")
                return {
                    "specialist": "forecast",
                    "status": "error",
                    "error": "Failed to fetch historical price data"
                }
            
            # Get company fundamentals for context
            print(f"  Fetching company fundamentals...")
            company_result = self.yahoo_api.get_company_info(symbol)
            
            # Generate forecast using LLM
            print(f"  Generating forecast with LLM...")
            forecast_analysis = self._generate_forecast_with_llm(
                price_result, historical_data, company_result, symbol
            )
            
            # Display findings
            print(f"  Current Price: ${forecast_analysis.get('current_price', 0):.2f}")
            print(f"  Historical Trend: {forecast_analysis.get('trend_direction', 'N/A')}")
            print(f"  6-Month Change: {forecast_analysis.get('historical_change_percent', 0):.2f}%")
            print(f"  Volatility: {forecast_analysis.get('volatility', 0):.2f}%")
            if forecast_analysis.get('forecast_price'):
                print(f"  Forecast Price (1 month): ${forecast_analysis.get('forecast_price', 0):.2f}")
                print(f"  Forecast Change: {forecast_analysis.get('forecast_change_percent', 0):.2f}%")
            if forecast_analysis.get('llm_insights'):
                llm_insights = forecast_analysis.get('llm_insights', '')
                print(f"  Forecast Insights:")
                # Print full insights with proper indentation
                for line in llm_insights.split('\n'):
                    print(f"    {line}")
            
            return {
                "specialist": "forecast",
                "status": "success",
                "analysis": {
                    "forecast_data": forecast_analysis,
                    "historical_data": historical_data,
                    "current_price": price_result.get("current_price", 0) if price_result.get("status") == "success" else 0
                }
            }
            
        except Exception as e:
            return {
                "specialist": "forecast",
                "status": "error",
                "error": str(e)
            }
    
    def _generate_forecast_with_llm(self, price_data: Dict[str, Any], historical_data: Dict[str, Any], 
                                    company_data: Dict[str, Any], symbol: str) -> Dict[str, Any]:
        """Use LLM to intelligently generate forecasts based on historical data and trends."""
        current_price = price_data.get("current_price", 0) if price_data.get("status") == "success" else 0
        trend = historical_data.get("trend_direction", "N/A")
        historical_change = historical_data.get("total_change_percent", 0)
        volatility = historical_data.get("volatility", 0)
        min_price = historical_data.get("min_price", 0)
        max_price = historical_data.get("max_price", 0)
        avg_price = historical_data.get("avg_price", 0)
        recent_prices = historical_data.get("recent_prices", [])
        
        company_name = company_data.get("company_name", "N/A") if company_data.get("status") == "success" else "N/A"
        sector = company_data.get("sector", "N/A") if company_data.get("status") == "success" else "N/A"
        
        # Build base forecast
        forecast = {
            "current_price": current_price,
            "trend_direction": trend,
            "historical_change_percent": historical_change,
            "volatility": volatility,
            "min_price_6mo": min_price,
            "max_price_6mo": max_price,
            "avg_price_6mo": avg_price
        }
        
        # Use LLM for intelligent forecasting
        try:
            forecast_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial forecasting expert. Analyze historical price trends and generate forward-looking forecasts.

Given historical price data and trends, provide:
1. Short-term forecast (1 month): Expected price range and direction
2. Trend continuation analysis: Will current trend continue or reverse?
3. Key support and resistance levels based on historical data
4. Risk assessment: Volatility and potential downside
5. Confidence level for the forecast

Consider:
- Historical trend direction and strength
- Volatility indicates price stability/instability
- Support (min) and resistance (max) levels from historical data
- Sector and company context
- Recent price movements (last 10 prices)
- Market conditions and momentum

Provide a realistic forecast with reasoning, not just optimistic predictions."""),
                ("human", """Generate financial forecast for {symbol}:

Company: {company_name}
Sector: {sector}
Current Price: ${current_price:.2f}

Historical Data (6 months):
- Trend Direction: {trend_direction}
- Total Change: {historical_change:.2f}%
- Volatility: {volatility:.2f}%
- Price Range: ${min_price:.2f} - ${max_price:.2f}
- Average Price: ${avg_price:.2f}
- Recent Prices (last 10): {recent_prices}

Provide a 1-month price forecast with:
1. Expected price range
2. Forecast direction (bullish/bearish/neutral)
3. Confidence level
4. Key factors driving the forecast
5. Risk considerations""")
            ])
            
            chain = forecast_prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "company_name": company_name,
                "sector": sector,
                "current_price": current_price,
                "trend_direction": trend,
                "historical_change": historical_change,
                "volatility": volatility,
                "min_price": min_price,
                "max_price": max_price,
                "avg_price": avg_price,
                "recent_prices": ", ".join([f"${p:.2f}" for p in recent_prices[-10:]])
            })
            
            llm_response = response.content.strip()
            
            # Extract forecast price from LLM response (try to find a price prediction)
            import re
            price_matches = re.findall(r'\$?(\d+\.?\d*)', llm_response)
            if price_matches and current_price > 0:
                # Try to find a reasonable forecast price (within 50% of current)
                for match in price_matches:
                    try:
                        candidate_price = float(match)
                        if 0.5 * current_price <= candidate_price <= 1.5 * current_price:
                            forecast["forecast_price"] = candidate_price
                            forecast["forecast_change_percent"] = ((candidate_price - current_price) / current_price) * 100
                            break
                    except:
                        continue
            
            # Extract forecast direction
            llm_lower = llm_response.lower()
            if "bullish" in llm_lower or "upward" in llm_lower or "rise" in llm_lower:
                forecast["forecast_direction"] = "bullish"
            elif "bearish" in llm_lower or "downward" in llm_lower or "decline" in llm_lower:
                forecast["forecast_direction"] = "bearish"
            else:
                forecast["forecast_direction"] = "neutral"
            
            forecast["llm_insights"] = llm_response
            
        except Exception as e:
            print(f"  LLM forecast error: {e}, using fallback")
            # Fallback to trend-based forecast
            forecast = self._fallback_forecast(current_price, historical_data, forecast)
        
        return forecast
    
    def _fallback_forecast(self, current_price: float, historical_data: Dict[str, Any], 
                          base_forecast: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback forecast based on trend continuation if LLM fails."""
        trend = historical_data.get("trend_direction", "sideways")
        historical_change = historical_data.get("total_change_percent", 0)
        volatility = historical_data.get("volatility", 0)
        
        # Simple trend continuation forecast
        if trend == "uptrend" and historical_change > 0:
            # Continue upward trend but with reduced momentum
            forecast_change = historical_change * 0.3  # 30% of historical momentum
            forecast["forecast_price"] = current_price * (1 + forecast_change / 100)
            forecast["forecast_direction"] = "bullish"
            forecast["forecast_change_percent"] = forecast_change
        elif trend == "downtrend" and historical_change < 0:
            # Continue downward trend but with reduced momentum
            forecast_change = historical_change * 0.3
            forecast["forecast_price"] = current_price * (1 + forecast_change / 100)
            forecast["forecast_direction"] = "bearish"
            forecast["forecast_change_percent"] = forecast_change
        else:
            # Sideways or insufficient data - assume minimal change
            forecast["forecast_price"] = current_price
            forecast["forecast_direction"] = "neutral"
            forecast["forecast_change_percent"] = 0.0
        
        forecast["llm_insights"] = f"Trend-based forecast: {trend} with {volatility:.2f}% volatility"
        
        return forecast

