"""
Earnings Specialist Agent for Financial Analysis
"""

from typing import Dict, Any
from tools.data_sources import YahooFinanceAPI

class EarningsSpecialistAgent:
    """Specialist agent for earnings and financial analysis."""
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze earnings and financial data for a stock."""
        print(f"Earnings Specialist: Analyzing {symbol}")
        
        try:
            # Get financial data
            company_result = self.yahoo_api.get_company_info(symbol)
            price_result = self.yahoo_api.get_stock_price(symbol)
            
            if company_result.get("status") != "success":
                return {
                    "specialist": "earnings",
                    "status": "error",
                    "error": "Failed to fetch company data"
                }
            
            # Analyze financial metrics
            financial_analysis = self._analyze_financials(company_result, price_result)
            
            return {
                "specialist": "earnings",
                "status": "success",
                "analysis": {
                    "financial_metrics": financial_analysis,
                    "company_info": company_result
                }
            }
            
        except Exception as e:
            return {
                "specialist": "earnings",
                "status": "error",
                "error": str(e)
            }
    
    def _analyze_financials(self, company_data: Dict[str, Any], price_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze financial metrics."""
        analysis = {
            "company_name": company_data.get("company_name", "N/A"),
            "sector": company_data.get("sector", "N/A"),
            "industry": company_data.get("industry", "N/A"),
            "market_cap": company_data.get("market_cap", 0),
            "pe_ratio": company_data.get("pe_ratio", "N/A"),
            "current_price": price_data.get("current_price", 0),
            "price_change": price_data.get("change_percent", 0)
        }
        
        # Basic financial assessment
        if analysis["pe_ratio"] != "N/A" and analysis["pe_ratio"] > 0:
            if analysis["pe_ratio"] < 15:
                analysis["valuation_assessment"] = "Potentially undervalued"
            elif analysis["pe_ratio"] > 25:
                analysis["valuation_assessment"] = "Potentially overvalued"
            else:
                analysis["valuation_assessment"] = "Fairly valued"
        else:
            analysis["valuation_assessment"] = "Insufficient data"
        
        return analysis
