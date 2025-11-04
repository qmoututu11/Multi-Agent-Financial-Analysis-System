"""
Evaluator-Optimizer Workflow: Generate analysis → evaluate quality → refine using feedback
Uses LLM for intelligent evaluation and optimization
"""

from typing import Dict, Any, Optional
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from tools.data_sources import YahooFinanceAPI
from config import Config

class QualityEvaluator:
    """Evaluates the quality of analysis outputs using LLM intelligence."""
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
    def evaluate_analysis(self, analysis: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM to intelligently evaluate the quality of an analysis."""
        if not analysis:
            return {
                "overall_score": 0.0,
                "feedback": ["Analysis is empty"],
                "recommendations": ["Provide a complete analysis"],
                "evaluation_timestamp": datetime.now().isoformat()
            }
        
        evaluation_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert financial analysis quality evaluator. Evaluate investment analysis reports for quality and completeness.

Evaluate the analysis on these dimensions:
1. **Completeness**: Does it cover essential aspects (market data, fundamentals, sentiment, risks)?
2. **Accuracy**: Are data points and metrics accurate and well-cited?
3. **Clarity**: Is the analysis clear, well-structured, and easy to understand?
4. **Actionability**: Does it provide clear investment guidance and recommendations?
5. **Depth**: Does it go beyond surface-level information?

Provide:
- Overall quality score (0.0 to 1.0)
- Specific strengths (what's good about the analysis)
- Specific weaknesses and improvement recommendations
- Market impact assessment

Respond in JSON format:
{{"overall_score": 0.85, "strengths": ["strength1", "strength2"], "weaknesses": ["weakness1"], "recommendations": ["rec1", "rec2"], "completeness_score": 0.8, "accuracy_score": 0.9, "clarity_score": 0.85, "actionability_score": 0.75, "depth_score": 0.8}}"""),
            ("human", """Evaluate this financial analysis for {symbol}:

{analysis}

Provide a comprehensive quality evaluation with specific feedback.""")
        ])
        
        try:
            chain = evaluation_prompt | self.llm
            response = chain.invoke({
                "symbol": context.get("symbol", "STOCK"),
                "analysis": analysis
            })
            
            # Parse LLM response
            import json
            try:
                evaluation_result = json.loads(response.content.strip())
                
                # Ensure all required fields
                return {
                    "overall_score": evaluation_result.get("overall_score", 0.5),
                    "feedback": evaluation_result.get("strengths", []),
                    "recommendations": evaluation_result.get("recommendations", []),
                    "weaknesses": evaluation_result.get("weaknesses", []),
                    "completeness_score": evaluation_result.get("completeness_score", 0.5),
                    "accuracy_score": evaluation_result.get("accuracy_score", 0.5),
                    "clarity_score": evaluation_result.get("clarity_score", 0.5),
                    "actionability_score": evaluation_result.get("actionability_score", 0.5),
                    "depth_score": evaluation_result.get("depth_score", 0.5),
                    "evaluation_timestamp": datetime.now().isoformat()
                }
            except json.JSONDecodeError:
                # Fallback parsing
                return self._parse_evaluation_response(response.content, analysis)
                
        except Exception as e:
            print(f"LLM evaluation error: {e}, using fallback")
            return self._fallback_evaluation(analysis)
    
    def _parse_evaluation_response(self, response_text: str, analysis: str) -> Dict[str, Any]:
        """Parse evaluation response if not valid JSON."""
        # Extract score (look for numbers between 0 and 1)
        import re
        score_match = re.search(r'0\.\d+|1\.0', response_text)
        score = float(score_match.group()) if score_match else 0.5
        
        # Extract strengths and recommendations
        strengths = []
        recommendations = []
        
        if "strength" in response_text.lower():
            # Simple extraction
            strengths.append("LLM identified strengths")
        
        if "recommend" in response_text.lower() or "improve" in response_text.lower():
            recommendations.append("LLM suggested improvements")
        
        return {
            "overall_score": score,
            "feedback": strengths,
            "recommendations": recommendations,
            "weaknesses": [],
            "evaluation_timestamp": datetime.now().isoformat()
        }
    
    def _fallback_evaluation(self, analysis: str) -> Dict[str, Any]:
        """Fallback keyword-based evaluation if LLM fails."""
        score = 0.5
        feedback = []
        recommendations = []
        
        if "recommendation" in analysis.lower():
            score += 0.1
            feedback.append("Includes investment recommendation")
        else:
            recommendations.append("Include investment recommendation")
        
        if "risk" in analysis.lower():
            score += 0.1
            feedback.append("Addresses risk factors")
        else:
            recommendations.append("Include risk assessment")
        
        if len(analysis) > 200:
            score += 0.1
        
        return {
            "overall_score": min(score, 1.0),
            "feedback": feedback,
            "recommendations": recommendations,
            "weaknesses": [],
            "evaluation_timestamp": datetime.now().isoformat()
        }

class AnalysisOptimizer:
    """Optimizes analysis based on evaluation feedback using LLM."""
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
    def optimize_analysis(self, analysis: str, evaluation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Use LLM to intelligently optimize analysis based on evaluation feedback.
        
        This method gathers additional information when weaknesses are identified,
        then uses LLM to incorporate the new data into an improved analysis.
        """
        if not analysis:
            return "No analysis to optimize"
        
        symbol = context.get("symbol", "STOCK")
        recommendations = evaluation.get("recommendations", [])
        weaknesses = evaluation.get("weaknesses", [])
        overall_score = evaluation.get("overall_score", 0.0)
        
        # Gather additional information based on identified weaknesses/recommendations
        additional_data = self._gather_additional_data(symbol, recommendations, weaknesses, analysis)
        
        optimization_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial analysis optimizer. Improve investment analyses based on quality evaluation feedback.

Your task:
1. Review the original analysis
2. Understand the evaluation feedback (strengths, weaknesses, recommendations)
3. Use the additional data provided to fill gaps and enhance accuracy
4. Enhance the analysis to address weaknesses and implement recommendations
5. Maintain the strengths while improving areas that need work
6. Ensure the improved analysis is more complete, accurate, clear, and actionable

Return an improved version of the analysis that:
- Addresses all recommendations
- Maintains existing strengths
- Incorporates the additional data provided
- Improves clarity and structure
- Adds missing elements (data, recommendations, risk assessment)
- Is more comprehensive and actionable"""),
            ("human", """Improve this financial analysis for {symbol}:

ORIGINAL ANALYSIS:
{analysis}

QUALITY EVALUATION:
- Overall Score: {score}/1.0
- Strengths: {strengths}
- Weaknesses: {weaknesses}
- Recommendations: {recommendations}

ADDITIONAL DATA GATHERED:
{additional_data}

Provide an enhanced, improved version of the analysis that:
1. Incorporates the additional data to address identified weaknesses
2. Maintains the original strengths
3. Implements the recommendations
4. Is more complete, accurate, and actionable""")
        ])
        
        try:
            chain = optimization_prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "analysis": analysis,
                "score": overall_score,
                "strengths": ", ".join(evaluation.get("feedback", [])),
                "weaknesses": ", ".join(weaknesses) if weaknesses else "None identified",
                "recommendations": ", ".join(recommendations) if recommendations else "None",
                "additional_data": additional_data if additional_data else "No additional data gathered (analysis already comprehensive)"
            })
            
            optimized_analysis = response.content.strip()
            
            # Add evaluation metadata
            result = f"""IMPROVED ANALYSIS FOR {symbol.upper()}
{'=' * 50}

QUALITY EVALUATION SUMMARY:
- Previous Score: {overall_score:.2f}/1.0
- Key Improvements Made:
"""
            for rec in recommendations[:5]:  # Top 5 recommendations
                result += f"  • {rec}\n"
            
            if additional_data:
                result += f"\nAdditional Data Gathered: Yes\n"
            
            result += f"\n{optimized_analysis}"
            
            return result
            
        except Exception as e:
            print(f"LLM optimization error: {e}, using fallback")
            return self._fallback_optimize(analysis, evaluation, context)
    
    def _gather_additional_data(self, symbol: str, recommendations: list, weaknesses: list, current_analysis: str) -> str:
        """Gather additional information based on identified weaknesses and recommendations.
        
        This is what makes iterations improve - we actually fetch more data when needed.
        """
        gathered_data = []
        recommendation_text = " ".join(recommendations).lower() + " " + " ".join(weaknesses).lower()
        analysis_lower = current_analysis.lower()
        
        # Check if we need financial/price data
        needs_price_data = any(keyword in recommendation_text or keyword in analysis_lower 
                              for keyword in ["price", "financial data", "market data", "valuation", "metrics", "financial"])
        if not needs_price_data:
            # Check if analysis already has price data
            if "price" not in analysis_lower and "$" not in current_analysis:
                needs_price_data = True
        
        if needs_price_data:
            try:
                price_data = self.yahoo_api.get_stock_price(symbol)
                if price_data.get("status") == "success":
                    gathered_data.append(f"PRICE DATA:\n- Current Price: ${price_data.get('current_price', 'N/A')}")
                    gathered_data.append(f"- Day Change: {price_data.get('change_percent', 'N/A')}%")
                    gathered_data.append(f"- Volume: {price_data.get('volume', 'N/A'):,}")
            except Exception as e:
                print(f"Error gathering price data: {e}")
        
        # Check if we need company fundamentals
        needs_fundamentals = any(keyword in recommendation_text 
                                for keyword in ["fundamentals", "company info", "sector", "industry", "pe ratio", "market cap"])
        if not needs_fundamentals:
            if "pe ratio" not in analysis_lower and "market cap" not in analysis_lower:
                needs_fundamentals = True
        
        if needs_fundamentals:
            try:
                company_data = self.yahoo_api.get_company_info(symbol)
                if company_data.get("status") == "success":
                    gathered_data.append(f"\nCOMPANY FUNDAMENTALS:")
                    gathered_data.append(f"- Company: {company_data.get('company_name', 'N/A')}")
                    gathered_data.append(f"- Sector: {company_data.get('sector', 'N/A')}")
                    gathered_data.append(f"- Industry: {company_data.get('industry', 'N/A')}")
                    if company_data.get('pe_ratio') != 'N/A':
                        gathered_data.append(f"- P/E Ratio: {company_data.get('pe_ratio', 'N/A')}")
                    if company_data.get('market_cap'):
                        market_cap_b = company_data.get('market_cap', 0) / 1e9
                        gathered_data.append(f"- Market Cap: ${market_cap_b:.2f}B")
            except Exception as e:
                print(f"Error gathering company data: {e}")
        
        # Check if we need news data
        needs_news = any(keyword in recommendation_text 
                        for keyword in ["news", "sentiment", "recent events", "market sentiment", "latest developments"])
        if not needs_news:
            if "news" not in analysis_lower and "recent" not in analysis_lower:
                needs_news = True
        
        if needs_news:
            try:
                news_data = self.yahoo_api.get_news(symbol, max_articles=5)
                if news_data.get("status") == "success" and news_data.get("articles"):
                    gathered_data.append(f"\nRECENT NEWS ({news_data.get('count', 0)} articles):")
                    for article in news_data.get("articles", [])[:3]:
                        title = article.get("title", "N/A")
                        summary = article.get("summary", "")[:200]
                        if summary:
                            gathered_data.append(f"- {title}: {summary}...")
                        else:
                            gathered_data.append(f"- {title}")
            except Exception as e:
                print(f"Error gathering news data: {e}")
        
        return "\n".join(gathered_data) if gathered_data else ""
    
    def _fallback_optimize(self, analysis: str, evaluation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Fallback optimization if LLM fails."""
        symbol = context.get("symbol", "STOCK")
        recommendations = evaluation.get("recommendations", [])
        
        optimized_parts = []
        optimized_parts.append(f"OPTIMIZED ANALYSIS FOR {symbol.upper()}")
        optimized_parts.append("=" * 50)
        
        # Add missing data
        if any("financial data" in rec.lower() for rec in recommendations):
            try:
                price_data = self.yahoo_api.get_stock_price(symbol)
                if price_data.get("status") == "success":
                    optimized_parts.append(f"\nAdditional Financial Data:")
                    optimized_parts.append(f"Current Price: ${price_data.get('current_price', 'N/A')}")
                    optimized_parts.append(f"Day Change: {price_data.get('change_percent', 'N/A')}%")
            except:
                pass
        
        optimized_parts.append(f"\n{analysis}")
        
        # Add recommendations
        if recommendations:
            optimized_parts.append(f"\nImprovements Based on Evaluation:")
            for rec in recommendations[:3]:
                optimized_parts.append(f"  • {rec}")
        
        return "\n".join(optimized_parts)

class EvaluatorOptimizerWorkflow:
    """
    Evaluator-Optimizer Workflow: Generate → Evaluate → Refine
    Can also evaluate and optimize results from other workflows
    """
    
    def __init__(self):
        self.evaluator = QualityEvaluator()
        self.optimizer = AnalysisOptimizer()
        self.yahoo_api = YahooFinanceAPI()
    
    def execute_workflow(self, symbol: str, focus: str = "comprehensive", max_iterations: int = 3, 
                        initial_analysis: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the evaluator-optimizer workflow.
        
        Args:
            symbol: Stock symbol
            focus: Analysis focus
            max_iterations: Maximum optimization iterations
            initial_analysis: Optional pre-existing analysis to evaluate (if None, generates one)
        """
        print(f"\n Evaluator-Optimizer Workflow: {symbol}")
        print("=" * 40)
        
        context = {
            "symbol": symbol,
            "focus": focus,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Get initial analysis (generate or use provided)
            if initial_analysis:
                print("Step 1: Using provided analysis for evaluation...")
                initial_analysis_text = initial_analysis
            else:
                print("Step 1: Generating initial analysis...")
                initial_analysis_text = self._generate_analysis(symbol, focus)
            
            if not initial_analysis_text:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Failed to get initial analysis",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 2: Evaluate quality using LLM
            print("Step 2: Evaluating analysis quality with LLM...")
            evaluation = self.evaluator.evaluate_analysis(initial_analysis_text, context)
            print(f"  Initial Quality Score: {evaluation.get('overall_score', 0):.2f}/1.0")
            
            # Step 3: Optimize iteratively using LLM
            print("Step 3: Optimizing analysis with LLM...")
            current_analysis = initial_analysis_text
            optimization_history = []
            
            for iteration in range(max_iterations):
                print(f"  Iteration {iteration + 1}/{max_iterations}")
                
                # Evaluate current analysis
                current_evaluation = self.evaluator.evaluate_analysis(current_analysis, context)
                current_score = current_evaluation.get("overall_score", 0)
                
                print(f"    Current Score: {current_score:.2f}/1.0")
                
                # Check if quality is sufficient
                if current_score >= 0.8:
                    print(f"  Quality sufficient: {current_score:.2f}")
                    break
                
                # Identify what needs improvement
                weaknesses = current_evaluation.get("weaknesses", [])
                recommendations = current_evaluation.get("recommendations", [])
                if weaknesses or recommendations:
                    print(f"    Identified Issues: {len(weaknesses)} weaknesses, {len(recommendations)} recommendations")
                    print(f"    Gathering additional data to address weaknesses...")
                
                # Optimize using LLM (this will gather additional data if needed)
                optimized_analysis = self.optimizer.optimize_analysis(
                    current_analysis, current_evaluation, context
                )
                
                # Check if new data was gathered
                if "Additional Data Gathered: Yes" in optimized_analysis:
                    print(f"    New data gathered and incorporated into analysis")
                
                # Re-evaluate to see improvement
                new_evaluation = self.evaluator.evaluate_analysis(optimized_analysis, context)
                new_score = new_evaluation.get("overall_score", 0)
                improvement = new_score - current_score
                
                if improvement > 0:
                    print(f"    Score improved: {current_score:.2f} -> {new_score:.2f} (+{improvement:.2f})")
                else:
                    print(f"    Score: {current_score:.2f} -> {new_score:.2f}")
                
                optimization_history.append({
                    "iteration": iteration + 1,
                    "score_before": current_score,
                    "score_after": new_score,
                    "improvement": improvement,
                    "evaluation": current_evaluation,
                    "analysis_preview": optimized_analysis[:200] + "..." if len(optimized_analysis) > 200 else optimized_analysis
                })
                
                current_analysis = optimized_analysis
            
            # Final evaluation
            print("Step 4: Final evaluation...")
            final_evaluation = self.evaluator.evaluate_analysis(current_analysis, context)
            print(f"  Final Quality Score: {final_evaluation.get('overall_score', 0):.2f}/1.0")
            
            return {
                "symbol": symbol,
                "focus": focus,
                "status": "success",
                "workflow": "evaluator_optimizer",
                "initial_analysis": initial_analysis_text,
                "final_analysis": current_analysis,
                "initial_evaluation": evaluation,
                "final_evaluation": final_evaluation,
                "optimization_history": optimization_history,
                "iterations_completed": len(optimization_history),
                "quality_improvement": final_evaluation.get("overall_score", 0) - evaluation.get("overall_score", 0),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Workflow error: {str(e)}")
            return {
                "symbol": symbol,
                "focus": focus,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def evaluate_external_analysis(self, analysis: str, symbol: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate an analysis from another workflow/agent."""
        eval_context = context or {"symbol": symbol}
        return self.evaluator.evaluate_analysis(analysis, eval_context)
    
    def _generate_analysis(self, symbol: str, focus: str) -> str:
        """Generate initial analysis for the symbol."""
        try:
            # Get basic data
            price_data = self.yahoo_api.get_stock_price(symbol)
            company_data = self.yahoo_api.get_company_info(symbol)
            
            analysis_parts = []
            analysis_parts.append(f"ANALYSIS FOR {symbol.upper()}")
            analysis_parts.append("=" * 30)
            
            # Price information
            if price_data.get("status") == "success":
                analysis_parts.append(f"Current Price: ${price_data.get('current_price', 'N/A')}")
                analysis_parts.append(f"Day Change: {price_data.get('change_percent', 'N/A')}%")
            
            # Company information
            if company_data.get("status") == "success":
                analysis_parts.append(f"Company: {company_data.get('company_name', 'N/A')}")
                analysis_parts.append(f"Sector: {company_data.get('sector', 'N/A')}")
                analysis_parts.append(f"Industry: {company_data.get('industry', 'N/A')}")
            
            # Focus-specific analysis
            if focus == "earnings":
                analysis_parts.append("\nEARNINGS FOCUS:")
                analysis_parts.append("This analysis focuses on financial performance and earnings trends.")
            elif focus == "news":
                analysis_parts.append("\nNEWS FOCUS:")
                analysis_parts.append("This analysis focuses on news sentiment and market trends.")
            elif focus == "technical":
                analysis_parts.append("\nTECHNICAL FOCUS:")
                analysis_parts.append("This analysis focuses on technical indicators and price trends.")
            else:
                analysis_parts.append("\nCOMPREHENSIVE FOCUS:")
                analysis_parts.append("This analysis covers multiple aspects of the investment.")
            
            # Basic recommendation
            analysis_parts.append(f"\nRECOMMENDATION:")
            analysis_parts.append(f"Based on the available data for {symbol}, consider the following factors:")
            analysis_parts.append("- Current market conditions")
            analysis_parts.append("- Company fundamentals")
            analysis_parts.append("- Risk tolerance")
            analysis_parts.append("- Investment timeline")
            
            return "\n".join(analysis_parts)
            
        except Exception as e:
            return f"Error generating analysis: {str(e)}"
