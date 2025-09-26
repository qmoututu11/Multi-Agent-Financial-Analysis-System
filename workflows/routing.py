"""
Routing Workflow: Direct content to the right specialist (earnings, news, or market analyzers)
"""

from typing import Dict, Any, List
from datetime import datetime
from enum import Enum

from agents.specialist_agents import NewsSpecialistAgent, EarningsSpecialistAgent, MarketSpecialistAgent

class SpecialistType(Enum):
    """Types of specialist agents."""
    NEWS = "news"
    EARNINGS = "earnings"
    MARKET = "market"
    COMPREHENSIVE = "comprehensive"

class RoutingWorkflow:
    """
    Routing Workflow: Direct content to the right specialist
    """
    
    def __init__(self):
        self.news_specialist = NewsSpecialistAgent()
        self.earnings_specialist = EarningsSpecialistAgent()
        self.market_specialist = MarketSpecialistAgent()
    
    def route_research_request(self, symbol: str, focus: str = "comprehensive") -> List[SpecialistType]:
        """Route a research request to appropriate specialists."""
        focus_lower = focus.lower()
        
        if focus_lower == "news":
            return [SpecialistType.NEWS]
        elif focus_lower == "earnings":
            return [SpecialistType.EARNINGS]
        elif focus_lower == "technical" or focus_lower == "market":
            return [SpecialistType.MARKET]
        else:  # comprehensive
            return [SpecialistType.NEWS, SpecialistType.EARNINGS, SpecialistType.MARKET]
    
    def execute_specialist_analysis(self, symbol: str, specialists: List[SpecialistType]) -> Dict[str, Any]:
        """Execute analysis using selected specialists."""
        print(f"\nRouting Workflow: {symbol}")
        print("=" * 40)
        
        results = {
            "symbol": symbol,
            "specialists_used": [s.value for s in specialists],
            "analyses": {},
            "combined_summary": "",
            "status": "success",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Execute each specialist
            for specialist_type in specialists:
                print(f"Executing {specialist_type.value} specialist...")
                
                if specialist_type == SpecialistType.NEWS:
                    analysis = self.news_specialist.analyze(symbol)
                elif specialist_type == SpecialistType.EARNINGS:
                    analysis = self.earnings_specialist.analyze(symbol)
                elif specialist_type == SpecialistType.MARKET:
                    analysis = self.market_specialist.analyze(symbol)
                else:
                    continue
                
                # Store analysis results
                if analysis.get("status") == "success":
                    results["analyses"][specialist_type.value] = analysis
                    print(f"{specialist_type.value.title()} analysis completed successfully")
                else:
                    print(f"{specialist_type.value.title()} analysis failed: {analysis.get('error', 'Unknown error')}")
                    results["analyses"][specialist_type.value] = analysis
            
            # Generate combined summary
            results["combined_summary"] = self._generate_combined_summary(results["analyses"])
            
            return results
            
        except Exception as e:
            print(f"Routing workflow error: {str(e)}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def _generate_combined_summary(self, analyses: Dict[str, Dict[str, Any]]) -> str:
        """Generate a combined summary from all specialist analyses."""
        if not analyses:
            return "No analyses available."
        
        summary_parts = []
        
        # News analysis summary
        if "news" in analyses and analyses["news"].get("status") == "success":
            news_data = analyses["news"].get("analysis", {})
            sentiment = news_data.get("sentiment", {})
            summary_parts.append(f"News sentiment: {sentiment.get('overall', 'neutral')} (confidence: {sentiment.get('confidence', 0):.2f})")
        
        # Earnings analysis summary
        if "earnings" in analyses and analyses["earnings"].get("status") == "success":
            earnings_data = analyses["earnings"].get("analysis", {})
            financial_metrics = earnings_data.get("financial_metrics", {})
            company_name = financial_metrics.get("company_name", "Unknown")
            valuation = financial_metrics.get("valuation_assessment", "Unknown")
            summary_parts.append(f"{company_name} valuation: {valuation}")
        
        # Market analysis summary
        if "market" in analyses and analyses["market"].get("status") == "success":
            market_data = analyses["market"].get("analysis", {})
            price_trends = market_data.get("price_trends", {})
            trend = price_trends.get("trend", "Unknown")
            price = price_trends.get("current_price", 0)
            summary_parts.append(f"Market trend: {trend} (Price: ${price:.2f})")
        
        if summary_parts:
            return " | ".join(summary_parts)
        else:
            return "Analysis completed but no detailed summaries available."