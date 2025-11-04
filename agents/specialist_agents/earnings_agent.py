"""
Earnings Specialist Agent for Financial Analysis

Analyzes company financials, valuation metrics, and earnings data.
Uses LLM for intelligent valuation assessment that considers sector/industry context,
not just simple rule-based thresholds. Falls back to rule-based analysis if LLM fails.
"""

from typing import Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from tools.data_sources import YahooFinanceAPI, SECEdgarAPI
from config import Config

class EarningsSpecialistAgent:
    """
    Specialist agent for earnings and financial analysis using LLM intelligence.
    
    Analyzes:
    - Company fundamentals (P/E ratio, market cap, sector, industry)
    - Valuation assessment (undervalued/fairly valued/overvalued)
    - Financial health insights
    
    Uses LLM to provide context-aware analysis considering sector averages
    and market conditions, not just fixed thresholds.
    """
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
        self.sec_api = SECEdgarAPI()  # SEC filings for official financial data
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
    def analyze(self, symbol: str) -> Dict[str, Any]:
        """Analyze earnings and financial data for a stock."""
        try:
            # Get financial data
            print(f"  Fetching financial data for {symbol}...")
            company_result = self.yahoo_api.get_company_info(symbol)
            price_result = self.yahoo_api.get_stock_price(symbol)
            
            if company_result.get("status") != "success":
                print(f"  Error: Failed to fetch company data")
                return {
                    "specialist": "earnings",
                    "status": "error",
                    "error": "Failed to fetch company data"
                }
            
            # Use price_result if available, otherwise use defaults
            if price_result.get("status") != "success":
                price_result = {"current_price": 0, "change_percent": 0}
            
            # Fetch SEC filings for additional context
            print(f"  Fetching SEC filings for additional context...")
            sec_filings = self._get_sec_filings_info(symbol)
            
            # Analyze financial metrics using LLM
            print(f"  Analyzing financials with LLM...")
            financial_analysis = self._analyze_financials_with_llm(company_result, price_result, symbol, sec_filings)
            
            # Display findings
            print(f"  Company: {financial_analysis.get('company_name', 'N/A')}")
            print(f"  Sector: {financial_analysis.get('sector', 'N/A')}")
            print(f"  Industry: {financial_analysis.get('industry', 'N/A')}")
            print(f"  Current Price: ${financial_analysis.get('current_price', 0):.2f}")
            if financial_analysis.get('pe_ratio') != 'N/A':
                print(f"  P/E Ratio: {financial_analysis.get('pe_ratio', 'N/A')}")
                print(f"  Valuation: {financial_analysis.get('valuation_assessment', 'N/A')}")
            if financial_analysis.get('market_cap'):
                market_cap_b = financial_analysis.get('market_cap', 0) / 1e9
                print(f"  Market Cap: ${market_cap_b:.2f}B")
            print(f"  Price Change: {financial_analysis.get('price_change', 0):.2f}%")
            if financial_analysis.get('llm_insights'):
                llm_insights = financial_analysis.get('llm_insights', '')
                print(f"  LLM Insights:")
                # Print full insights with proper indentation
                for line in llm_insights.split('\n'):
                    print(f"    {line}")
            
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
    
    def _get_sec_filings_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get SEC filings information for additional context."""
        try:
            # Get most recent 10-K (annual report) and 10-Q (quarterly report)
            recent_10k = self.sec_api.get_filing_summary(symbol, "10-K")
            recent_10q = self.sec_api.get_filing_summary(symbol, "10-Q")
            
            sec_info = {
                "has_10k": recent_10k.get("status") == "success",
                "has_10q": recent_10q.get("status") == "success",
                "latest_10k_date": recent_10k.get("filing_date") if recent_10k.get("status") == "success" else None,
                "latest_10q_date": recent_10q.get("filing_date") if recent_10q.get("status") == "success" else None
            }
            
            if sec_info["has_10k"] or sec_info["has_10q"]:
                print(f"    Found SEC filings: 10-K ({sec_info['latest_10k_date'] or 'N/A'}), 10-Q ({sec_info['latest_10q_date'] or 'N/A'})")
            
            return sec_info
        except Exception as e:
            print(f"    SEC filings error: {e} (continuing without SEC data)")
            return None
    
    def _analyze_financials_with_llm(self, company_data: Dict[str, Any], price_data: Dict[str, Any], symbol: str, sec_filings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to intelligently analyze financial metrics and provide valuation assessment."""
        # Extract basic data
        company_name = company_data.get("company_name", "N/A")
        sector = company_data.get("sector", "N/A")
        industry = company_data.get("industry", "N/A")
        market_cap = company_data.get("market_cap", 0)
        pe_ratio = company_data.get("pe_ratio", "N/A")
        current_price = price_data.get("current_price", 0)
        price_change = price_data.get("change_percent", 0)
        
        # Build base analysis
        analysis = {
            "company_name": company_name,
            "sector": sector,
            "industry": industry,
            "market_cap": market_cap,
            "pe_ratio": pe_ratio,
            "current_price": current_price,
            "price_change": price_change
        }
        
        # Use LLM for intelligent valuation assessment
        try:
            # Build SEC filings context
            sec_context = ""
            if sec_filings:
                sec_context = f"\nSEC Filings Available:\n"
                if sec_filings.get("has_10k"):
                    sec_context += f"- Most recent 10-K (Annual Report): {sec_filings.get('latest_10k_date', 'N/A')}\n"
                if sec_filings.get("has_10q"):
                    sec_context += f"- Most recent 10-Q (Quarterly Report): {sec_filings.get('latest_10q_date', 'N/A')}\n"
                sec_context += "Note: SEC filings contain official financial statements and regulatory disclosures."
            else:
                sec_context = "\nSEC Filings: Not available (company may not be required to file or data unavailable)"
            
            valuation_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a financial analyst expert. Analyze stock valuation based on financial metrics.

Given the financial data, provide:
1. Valuation assessment (undervalued/fairly valued/overvalued)
2. Brief reasoning (consider sector, industry, market conditions)
3. Key insights about the financial health

Consider:
- P/E ratio relative to sector/industry averages
- Market cap indicates company size
- Price change reflects recent market sentiment
- Sector and industry context matters
- SEC filings provide official financial data and regulatory disclosures
- Recent filings (10-K, 10-Q) indicate active regulatory compliance and financial transparency"""),
                ("human", """Analyze valuation for {symbol}:

Company: {company_name}
Sector: {sector}
Industry: {industry}
Current Price: ${current_price}
P/E Ratio: {pe_ratio}
Market Cap: ${market_cap:,.0f}
Price Change: {price_change:.2f}%
{sec_filings_context}

Provide a concise valuation assessment and reasoning.""")
            ])
            
            chain = valuation_prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "company_name": company_name,
                "sector": sector,
                "industry": industry,
                "current_price": current_price,
                "pe_ratio": pe_ratio if pe_ratio != "N/A" else "N/A",
                "market_cap": market_cap,
                "price_change": price_change,
                "sec_filings_context": sec_context
            })
            
            llm_response = response.content.strip()
            
            # Extract valuation assessment from LLM response
            if "undervalued" in llm_response.lower():
                analysis["valuation_assessment"] = "Potentially undervalued"
            elif "overvalued" in llm_response.lower():
                analysis["valuation_assessment"] = "Potentially overvalued"
            else:
                analysis["valuation_assessment"] = "Fairly valued"
            
            analysis["llm_insights"] = llm_response
            
        except Exception as e:
            print(f"  LLM analysis error: {e}, using fallback")
            # Fallback to rule-based assessment
            analysis["valuation_assessment"] = self._fallback_valuation(pe_ratio)
            analysis["llm_insights"] = ""
        
        return analysis
    
    def _fallback_valuation(self, pe_ratio: Any) -> str:
        """Fallback rule-based valuation if LLM fails."""
        if pe_ratio != "N/A" and pe_ratio > 0:
            if pe_ratio < 15:
                return "Potentially undervalued"
            elif pe_ratio > 25:
                return "Potentially overvalued"
            else:
                return "Fairly valued"
        return "Insufficient data"
