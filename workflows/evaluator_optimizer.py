"""
Evaluator-Optimizer Workflow: Generate analysis → evaluate quality → refine using feedback
"""

from typing import Dict, Any
from datetime import datetime

from tools.data_sources import YahooFinanceAPI
from config import Config

class QualityEvaluator:
    """Evaluates the quality of analysis outputs."""
    
    def evaluate_analysis(self, analysis: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of an analysis."""
        if not analysis:
            return {
                "overall_score": 0.0,
                "feedback": ["Analysis is empty"],
                "recommendations": ["Provide a complete analysis"]
            }
        
        # Simple quality evaluation
        score = 0.5  # Base score
        feedback = []
        recommendations = []
        
        # Check for key elements
        if "recommendation" in analysis.lower():
            score += 0.1
            feedback.append(" Includes investment recommendation")
        else:
            recommendations.append("Include investment recommendation")
        
        if "risk" in analysis.lower():
            score += 0.1
            feedback.append(" Addresses risk factors")
        else:
            recommendations.append("Include risk assessment")
        
        if len(analysis) > 200:  # Substantial analysis
            score += 0.1
            feedback.append(" Provides comprehensive analysis")
        else:
            recommendations.append("Expand analysis with more details")
        
        if any(word in analysis.lower() for word in ["buy", "sell", "hold", "strong", "weak"]):
            score += 0.1
            feedback.append(" Provides clear investment guidance")
        else:
            recommendations.append("Provide clearer investment guidance")
        
        if any(char in analysis for char in ['$', '%', 'ratio']):
            score += 0.1
            feedback.append(" Includes specific financial metrics")
        else:
            recommendations.append("Include more specific financial data")
        
        return {
            "overall_score": min(score, 1.0),
            "feedback": feedback,
            "recommendations": recommendations,
            "evaluation_timestamp": datetime.now().isoformat()
        }

class AnalysisOptimizer:
    """Optimizes analysis based on evaluation feedback."""
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
    
    def optimize_analysis(self, analysis: str, evaluation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Optimize analysis based on evaluation feedback."""
        if not analysis:
            return "No analysis to optimize"
        
        symbol = context.get("symbol", "STOCK")
        recommendations = evaluation.get("recommendations", [])
        
        optimized_parts = []
        optimized_parts.append(f"OPTIMIZED ANALYSIS FOR {symbol.upper()}")
        optimized_parts.append("=" * 50)
        
        # Original analysis
        optimized_parts.append("\nORIGINAL ANALYSIS:")
        optimized_parts.append(analysis)
        
        # Evaluation summary
        overall_score = evaluation.get("overall_score", 0.0)
        optimized_parts.append(f"\nQUALITY EVALUATION:")
        optimized_parts.append(f"Overall Quality Score: {overall_score:.2f}/1.0")
        
        # Feedback
        feedback = evaluation.get("feedback", [])
        if feedback:
            optimized_parts.append(f"\nSTRENGTHS:")
            for item in feedback:
                optimized_parts.append(f"  {item}")
        
        # Recommendations
        if recommendations:
            optimized_parts.append(f"\nIMPROVEMENT RECOMMENDATIONS:")
            for i, rec in enumerate(recommendations, 1):
                optimized_parts.append(f"  {i}. {rec}")
        
        # Enhanced analysis
        optimized_parts.append(f"\nENHANCED ANALYSIS:")
        enhanced_analysis = self._enhance_analysis(analysis, evaluation, context)
        optimized_parts.append(enhanced_analysis)
        
        return "\n".join(optimized_parts)
    
    def _enhance_analysis(self, analysis: str, evaluation: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Enhance the analysis based on evaluation feedback."""
        symbol = context.get("symbol", "STOCK")
        recommendations = evaluation.get("recommendations", [])
        
        enhanced_parts = []
        
        # Add specific data if missing
        if "Include more specific financial data" in recommendations:
            try:
                price_data = self.yahoo_api.get_stock_price(symbol)
                if price_data.get("status") == "success":
                    enhanced_parts.append(f"Additional Financial Data for {symbol}:")
                    enhanced_parts.append(f"Current Price: ${price_data.get('current_price', 'N/A')}")
                    enhanced_parts.append(f"Day Change: {price_data.get('change_percent', 'N/A')}%")
                    enhanced_parts.append(f"Volume: {price_data.get('volume', 'N/A'):,}")
            except:
                pass
        
        # Add structure if missing
        if "Expand analysis with more details" in recommendations:
            enhanced_parts.append("\nSTRUCTURED ANALYSIS:")
            enhanced_parts.append("1. Current Situation:")
            enhanced_parts.append("2. Key Factors:")
            enhanced_parts.append("3. Risk Assessment:")
            enhanced_parts.append("4. Investment Recommendation:")
        
        # Add original analysis
        enhanced_parts.append(f"\n{analysis}")
        
        # Add recommendations if missing
        if "Include investment recommendation" in recommendations:
            enhanced_parts.append(f"\nINVESTMENT RECOMMENDATION:")
            enhanced_parts.append(f"Based on the analysis of {symbol}, consider the following:")
            enhanced_parts.append("- Monitor key financial metrics regularly")
            enhanced_parts.append("- Assess market conditions and news sentiment")
            enhanced_parts.append("- Consider risk tolerance and investment timeline")
        
        # Add risk assessment if missing
        if "Include risk assessment" in recommendations:
            enhanced_parts.append(f"\nRISK ASSESSMENT:")
            enhanced_parts.append(f"Key risks to consider for {symbol}:")
            enhanced_parts.append("- Market volatility and economic conditions")
            enhanced_parts.append("- Company-specific risks and challenges")
            enhanced_parts.append("- Industry trends and competitive landscape")
        
        return "\n".join(enhanced_parts)

class EvaluatorOptimizerWorkflow:
    """
    Evaluator-Optimizer Workflow: Generate → Evaluate → Refine
    """
    
    def __init__(self):
        self.evaluator = QualityEvaluator()
        self.optimizer = AnalysisOptimizer()
        self.yahoo_api = YahooFinanceAPI()
    
    def execute_workflow(self, symbol: str, focus: str = "comprehensive", max_iterations: int = 3) -> Dict[str, Any]:
        """Execute the evaluator-optimizer workflow."""
        print(f"\n Evaluator-Optimizer Workflow: {symbol}")
        print("=" * 40)
        
        context = {
            "symbol": symbol,
            "focus": focus,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Step 1: Generate initial analysis
            print("Step 1: Generating initial analysis...")
            initial_analysis = self._generate_analysis(symbol, focus)
            
            if not initial_analysis:
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Failed to generate initial analysis",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 2: Evaluate quality
            print("Step 2: Evaluating analysis quality...")
            evaluation = self.evaluator.evaluate_analysis(initial_analysis, context)
            
            # Step 3: Optimize iteratively
            print("Step 3: Optimizing analysis...")
            current_analysis = initial_analysis
            optimization_history = []
            
            for iteration in range(max_iterations):
                print(f"  Iteration {iteration + 1}/{max_iterations}")
                
                # Evaluate current analysis
                current_evaluation = self.evaluator.evaluate_analysis(current_analysis, context)
                
                # Check if quality is sufficient
                if current_evaluation.get("overall_score", 0) >= 0.8:
                    print(f"  Quality sufficient: {current_evaluation.get('overall_score', 0):.2f}")
                    break
                
                # Optimize
                optimized_analysis = self.optimizer.optimize_analysis(
                    current_analysis, current_evaluation, context
                )
                
                optimization_history.append({
                    "iteration": iteration + 1,
                    "score": current_evaluation.get("overall_score", 0),
                    "analysis": optimized_analysis
                })
                
                current_analysis = optimized_analysis
            
            # Final evaluation
            final_evaluation = self.evaluator.evaluate_analysis(current_analysis, context)
            
            return {
                "symbol": symbol,
                "focus": focus,
                "status": "success",
                "workflow": "evaluator_optimizer",
                "initial_analysis": initial_analysis,
                "final_analysis": current_analysis,
                "initial_evaluation": evaluation,
                "final_evaluation": final_evaluation,
                "optimization_history": optimization_history,
                "iterations_completed": len(optimization_history),
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
