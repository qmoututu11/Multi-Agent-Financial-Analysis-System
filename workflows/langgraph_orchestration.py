"""
LangGraph Orchestration Workflow
Multi-agent financial analysis with graph-based state management
"""

from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
 

from agents.specialist_agents import NewsSpecialistAgent, EarningsSpecialistAgent, MarketSpecialistAgent, ForecastSpecialistAgent
from config import Config


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow."""
    symbol: str
    focus: str
    workflow_type: str  # "comprehensive" (uses all specialist agents)
    use_evaluation: bool  # Whether to evaluate and optimize results
    
    # Agent outputs
    news_specialist_result: Dict[str, Any]
    earnings_specialist_result: Dict[str, Any]
    market_specialist_result: Dict[str, Any]
    forecast_specialist_result: Dict[str, Any]
    
    # Workflow results
    routing_result: Dict[str, Any]
    evaluator_optimizer_result: Dict[str, Any]
    
    # Combined results
    combined_result: Dict[str, Any]
    
    # Execution tracking
    nodes_executed: Annotated[List[str], operator.add]
    errors: Annotated[List[str], operator.add]
    timestamp: str


class LangGraphOrchestrator:
    """
    LangGraph-based orchestration for multi-agent financial analysis.
    Uses graph-based state management and conditional routing.
    """
    
    def __init__(self):
        """Initialize the LangGraph orchestrator."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Initialize agents
        # News specialist uses prompt chaining for enhanced LLM-powered analysis
        self.news_specialist = NewsSpecialistAgent(use_prompt_chaining=True)
        self.earnings_specialist = EarningsSpecialistAgent()
        self.market_specialist = MarketSpecialistAgent()
        self.forecast_specialist = ForecastSpecialistAgent()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("news_specialist", self._news_specialist_node)
        workflow.add_node("earnings_specialist", self._earnings_specialist_node)
        workflow.add_node("market_specialist", self._market_specialist_node)
        workflow.add_node("forecast_specialist", self._forecast_specialist_node)
        workflow.add_node("evaluator_optimizer", self._evaluator_optimizer_node)
        workflow.add_node("combine_results", self._combine_results_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "route")
        
        # Route to appropriate specialist(s) based on focus
        workflow.add_conditional_edges(
            "route",
            self._route_to_specialists,
            {
                "news": "news_specialist",
                "earnings": "earnings_specialist",
                "market": "market_specialist",
                "comprehensive": "news_specialist"
            }
        )
        
        # Conditional edges from specialists - route to next or skip to combine
        workflow.add_conditional_edges(
            "news_specialist",
            self._after_news_specialist,
            {
                "next": "earnings_specialist",
                "skip_earnings": "market_specialist",
                "combine": "combine_results"
            }
        )
        
        workflow.add_conditional_edges(
            "earnings_specialist",
            self._after_earnings_specialist,
            {
                "next": "market_specialist",
                "combine": "combine_results"
            }
        )
        
        workflow.add_conditional_edges(
            "market_specialist",
            self._after_market_specialist,
            {
                "next": "forecast_specialist",
                "combine": "combine_results"
            }
        )
        
        workflow.add_edge("forecast_specialist", "combine_results")
        
        # After combining results from routing/comprehensive, check if evaluation needed
        workflow.add_conditional_edges(
            "combine_results",
            self._should_evaluate_combined,
            {
                "evaluate": "evaluator_optimizer",
                "skip": "finalize"
            }
        )
        
        # Evaluator optimizer always goes to finalize
        workflow.add_edge("evaluator_optimizer", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _start_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow state."""
        print(f"\n LangGraph Orchestration: Starting analysis for {state['symbol']}")
        print("=" * 60)
        
        # Initialize state - nodes_executed uses operator.add so return new items
        return {
            **state,
            "nodes_executed": ["start"],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
            "news_specialist_result": {},
            "earnings_specialist_result": {},
            "market_specialist_result": {},
            "forecast_specialist_result": {},
            "routing_result": {},
            "evaluator_optimizer_result": {},
            "combined_result": {}
        }
    
    def _route_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize routing - determines which specialists to run based on focus."""
        symbol = state.get('symbol', 'unknown')
        focus = state.get('focus', 'comprehensive')
        
        print(f"\n Starting Analysis: {symbol}")
        print(f"Focus: {focus}")
        
        # Don't modify nodes_executed here - let route_node add itself
        return {
            **state,
            "workflow_type": "comprehensive",
            "nodes_executed": ["route"]
        }
    
    def _route_to_specialists(self, state: WorkflowState) -> str:
        """Route to first specialist based on focus."""
        focus = state.get('focus', 'comprehensive').lower()
        
        if focus == 'news':
            return 'news'
        elif focus == 'earnings':
            return 'earnings'
        elif focus in ['technical', 'market']:
            return 'market'
        elif focus == 'forecast':
            return 'market'  # Start with market, then forecast
        else:  # comprehensive
            return 'comprehensive'
    
    def _after_news_specialist(self, state: WorkflowState) -> str:
        """Determine next step after news specialist."""
        focus = state.get('focus', 'comprehensive').lower()
        
        if focus == 'news':
            # Only news requested, skip to combine
            return 'combine'
        else:  # comprehensive or other
            # Continue to earnings
            return 'next'
    
    def _after_earnings_specialist(self, state: WorkflowState) -> str:
        """Determine next step after earnings specialist."""
        focus = state.get('focus', 'comprehensive').lower()
        
        if focus == 'earnings':
            # Only earnings requested, skip to combine
            return 'combine'
        else:  # comprehensive or other
            # Continue to market
            return 'next'
    
    def _after_market_specialist(self, state: WorkflowState) -> str:
        """Determine next step after market specialist."""
        focus = state.get('focus', 'comprehensive').lower()
        
        if focus in ['technical', 'market']:
            # Only market requested, skip to combine
            return 'combine'
        elif focus == 'forecast':
            # Forecast requested, continue to forecast
            return 'next'
        else:  # comprehensive
            # Continue to forecast
            return 'next'
    
    def _should_evaluate_combined(self, state: WorkflowState) -> str:
        """Decide if combined routing results should be evaluated."""
        # Auto-evaluate combined results for quality improvement
        use_evaluation = state.get("use_evaluation", True)
        
        if use_evaluation:
            return "evaluate"
        else:
            return "skip"
    def _news_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the news specialist agent."""
        print(f"\n News Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.news_specialist.analyze(state['symbol'])
            return {
                **state,
                "news_specialist_result": result,
                "nodes_executed": ["news_specialist"]  # operator.add will merge automatically
            }
        except Exception as e:
            error_msg = f"News specialist error: {str(e)}"
            print(f" {error_msg}")
            return {
                **state,
                "news_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],  # operator.add will merge automatically
                "nodes_executed": ["news_specialist"]
            }
    
    def _earnings_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the earnings specialist agent."""
        print(f"\n Earnings Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.earnings_specialist.analyze(state['symbol'])
            return {
                **state,
                "earnings_specialist_result": result,
                "nodes_executed": ["earnings_specialist"]
            }
        except Exception as e:
            error_msg = f"Earnings specialist error: {str(e)}"
            print(f" {error_msg}")
            return {
                **state,
                "earnings_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["earnings_specialist"]
            }
    
    def _market_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the market specialist agent."""
        print(f"\n Market Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.market_specialist.analyze(state['symbol'])
            return {
                **state,
                "market_specialist_result": result,
                "nodes_executed": ["market_specialist"]
            }
        except Exception as e:
            error_msg = f"Market specialist error: {str(e)}"
            print(f" {error_msg}")
            return {
                **state,
                "market_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["market_specialist"]
            }
    
    def _forecast_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the forecast specialist agent."""
        print(f"\n Forecast Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.forecast_specialist.analyze(state['symbol'])
            return {
                **state,
                "forecast_specialist_result": result,
                "nodes_executed": ["forecast_specialist"]
            }
        except Exception as e:
            error_msg = f"Forecast specialist error: {str(e)}"
            print(f" {error_msg}")
            return {
                **state,
                "forecast_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["forecast_specialist"]
            }
    
    def _evaluator_optimizer_node(self, state: WorkflowState) -> WorkflowState:
        """Evaluate and optimize results from primary workflows."""
        print(f"\n Evaluator-Optimizer Node: Evaluating and optimizing results for {state['symbol']}")
        
        try:
            from workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
            
            # Get analysis from previous workflow(s)
            analysis_text = ""
            if state.get("combined_result"):
                # Use combined result from routing/comprehensive
                combined = state["combined_result"]
                if isinstance(combined, dict):
                    analysis_text = combined.get("summary", "")
                    if not analysis_text:
                        analysis_text = str(combined.get("result", ""))
            
            if not analysis_text:
                analysis_text = f"Analysis for {state['symbol']} completed."
            
            # Evaluate and optimize the analysis
            workflow = EvaluatorOptimizerWorkflow()
            result = workflow.execute_workflow(
                state['symbol'],
                state.get('focus', 'comprehensive'),
                max_iterations=2,
                initial_analysis=analysis_text
            )
            
            # Update combined result with optimized version
            if state.get("combined_result"):
                state["combined_result"]["optimized_analysis"] = result.get("final_analysis", "")
                state["combined_result"]["evaluation"] = result.get("final_evaluation", {})
            
            return {
                **state,
                "evaluator_optimizer_result": result,
                "nodes_executed": ["evaluator_optimizer"]
            }
        except Exception as e:
            error_msg = f"Evaluator-optimizer error: {str(e)}"
            print(f" {error_msg}")
            return {
                **state,
                "evaluator_optimizer_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["evaluator_optimizer"]
            }
    
    def _combine_results_node(self, state: WorkflowState) -> WorkflowState:
        """
        Combine results from all executed specialist nodes.
        
        Collects results from news, earnings, market, and forecast specialists based on focus,
        then generates a combined summary of all analyses.
        """
        print(f"\n Combining Results...")
        
        combined = {
            "symbol": state['symbol'],
            "focus": state.get('focus', 'comprehensive'),
            "workflow_type": "comprehensive",  # Always use comprehensive (all specialists)
            "nodes_executed": state.get('nodes_executed', []),
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Collect results from executed specialists only
        focus = state.get('focus', 'comprehensive').lower()
        
        if focus == 'news':
            combined["results"]["news"] = state.get("news_specialist_result", {})
        elif focus == 'earnings':
            combined["results"]["earnings"] = state.get("earnings_specialist_result", {})
        elif focus in ['technical', 'market']:
            combined["results"]["market"] = state.get("market_specialist_result", {})
        elif focus == 'forecast':
            combined["results"]["market"] = state.get("market_specialist_result", {})
            combined["results"]["forecast"] = state.get("forecast_specialist_result", {})
        else:  # comprehensive
            combined["results"]["news"] = state.get("news_specialist_result", {})
            combined["results"]["earnings"] = state.get("earnings_specialist_result", {})
            combined["results"]["market"] = state.get("market_specialist_result", {})
            combined["results"]["forecast"] = state.get("forecast_specialist_result", {})
        
        # Simple status check
        specialists_executed = [key for key, result in combined["results"].items() 
                               if isinstance(result, dict) and result.get("status") == "success"]
        print(f"  Successfully combined results from {len(specialists_executed)} specialist(s): {', '.join(specialists_executed)}")
        
        # Determine overall status
        errors = state.get("errors", [])
        if errors:
            combined["status"] = "partial_success" if combined["results"] else "error"
            combined["errors"] = errors
        else:
            combined["status"] = "success"
        
        # Generate summary
        combined["summary"] = self._generate_summary(combined)
        
        return {
            **state,
            "combined_result": combined,
            "nodes_executed": ["combine_results"]
        }
    
    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow."""
        print(f"\n Finalizing Results")
        print("=" * 60)
        
        combined = state.get("combined_result", {})
        nodes_executed = state.get("nodes_executed", [])
        symbol = state.get("symbol", "UNKNOWN")
        
        # Generate comprehensive analysis report
        analysis_report = self._generate_comprehensive_analysis(combined, state)
        print(analysis_report)
        
        # Note: Individual agent logs already shown detailed findings above
        # Note: Evaluator-optimizer results are automatically included in quality scoring
        
        return {
            **state,
            "combined_result": {
                **combined,
                "summary": combined.get("summary", ""),
                "comprehensive_analysis": analysis_report,
                "nodes_executed": nodes_executed  # Show actual execution path
            },
            "nodes_executed": ["finalize"]  # operator.add will merge
        }
    
    def _generate_comprehensive_analysis(self, combined: Dict[str, Any], state: WorkflowState) -> str:
        """Generate a comprehensive analysis report with synthesis and recommendations."""
        symbol = combined.get("symbol", "UNKNOWN")
        results = combined.get("results", {})
        
        # Extract data from all specialists
        earnings_data = results.get("earnings", {})
        market_data = results.get("market", {})
        news_data = results.get("news", {})
        forecast_data = results.get("forecast", {})
        
        report_parts = []
        report_parts.append(f"\n{'=' * 60}")
        report_parts.append(f"COMPREHENSIVE FINANCIAL ANALYSIS: {symbol.upper()}")
        report_parts.append(f"{'=' * 60}\n")
        
        # Key Metrics Summary
        report_parts.append("KEY METRICS:")
        report_parts.append("-" * 60)
        
        # Get current price from market or forecast
        current_price = 0
        if market_data.get("status") == "success":
            trends = market_data.get("analysis", {}).get("price_trends", {})
            current_price = trends.get('current_price', 0)
        elif forecast_data.get("status") == "success":
            forecast_analysis = forecast_data.get("analysis", {}).get("forecast_data", {})
            current_price = forecast_analysis.get('current_price', 0)
        
        if earnings_data.get("status") == "success":
            metrics = earnings_data.get("analysis", {}).get("financial_metrics", {})
            report_parts.append(f"Company: {metrics.get('company_name', 'N/A')} ({symbol.upper()})")
            report_parts.append(f"Sector/Industry: {metrics.get('sector', 'N/A')} / {metrics.get('industry', 'N/A')}")
            if current_price > 0:
                report_parts.append(f"Current Price: ${current_price:.2f}")
            if metrics.get('market_cap'):
                market_cap_b = metrics.get('market_cap', 0) / 1e9
                report_parts.append(f"Market Cap: ${market_cap_b:.2f}B")
            if metrics.get('pe_ratio') != 'N/A':
                report_parts.append(f"P/E Ratio: {metrics.get('pe_ratio', 'N/A')}")
        
        # Market Performance
        if market_data.get("status") == "success":
            trends = market_data.get("analysis", {}).get("price_trends", {})
            price_change = trends.get('price_change_percent', 0)
            trend = trends.get('trend', 'N/A')
            report_parts.append(f"Price Change: {price_change:+.2f}% | Trend: {trend.upper()}")
        
        report_parts.append("")
        
        # Synthesized Analysis Summary
        report_parts.append("SYNTHESIZED ANALYSIS:")
        report_parts.append("-" * 60)
        
        # Combine key insights from all specialists
        insights_summary = []
        
        # News sentiment
        if news_data.get("status") == "success":
            analysis = news_data.get("analysis", {})
            sentiment = analysis.get("sentiment", {})
            if isinstance(sentiment, dict):
                if "overall_sentiment" in sentiment:
                    overall = sentiment.get("overall_sentiment", "neutral").upper()
                    positive = sentiment.get("positive", 0)
                    negative = sentiment.get("negative", 0)
                    neutral = sentiment.get("neutral", 0)
                    insights_summary.append(f"News sentiment is {overall} ({positive} positive, {negative} negative, {neutral} neutral articles)")
                elif "overall" in sentiment:
                    overall = sentiment.get("overall", "neutral").upper()
                    insights_summary.append(f"News sentiment is {overall}")
        
        # Valuation
        if earnings_data.get("status") == "success":
            metrics = earnings_data.get("analysis", {}).get("financial_metrics", {})
            valuation = metrics.get("valuation_assessment", "")
            if valuation:
                insights_summary.append(f"Valuation: {valuation}")
        
        # Market trend
        if market_data.get("status") == "success":
            trends = market_data.get("analysis", {}).get("price_trends", {})
            trend = trends.get("trend", "N/A")
            volume_signal = trends.get("volume_signal", "N/A")
            insights_summary.append(f"Market trend: {trend.upper()} with {volume_signal.lower()} volume")
        
        # Forecast
        if forecast_data.get("status") == "success":
            forecast_analysis = forecast_data.get("analysis", {}).get("forecast_data", {})
            forecast_dir = forecast_analysis.get("forecast_direction", "neutral")
            forecast_change = forecast_analysis.get("forecast_change_percent", 0)
            if forecast_analysis.get('forecast_price'):
                forecast_price = forecast_analysis.get('forecast_price', 0)
                insights_summary.append(f"1-month forecast: ${forecast_price:.2f} ({forecast_change:+.2f}%) - {forecast_dir.upper()}")
        
        if insights_summary:
            for insight in insights_summary:
                report_parts.append(f"  • {insight}")
        
        report_parts.append("")
        
        # Investment Recommendations
        report_parts.append("INVESTMENT RECOMMENDATIONS:")
        report_parts.append("-" * 60)
        
        recommendations = []
        
        # Recommendation based on valuation
        if earnings_data.get("status") == "success":
            metrics = earnings_data.get("analysis", {}).get("financial_metrics", {})
            valuation = metrics.get("valuation_assessment", "")
            if "undervalued" in valuation.lower():
                recommendations.append("Consider BUY - Stock appears undervalued based on P/E ratio")
            elif "overvalued" in valuation.lower():
                recommendations.append("Consider CAUTION - Stock may be overvalued")
            else:
                recommendations.append("Consider HOLD - Stock appears fairly valued")
        
        # Recommendation based on market trend
        if market_data.get("status") == "success":
            trends = market_data.get("analysis", {}).get("price_trends", {})
            trend = trends.get("trend", "").lower()
            if "bullish" in trend:
                recommendations.append("Market momentum is BULLISH - Positive price action observed")
            elif "bearish" in trend:
                recommendations.append("Market momentum is BEARISH - Negative price action observed")
        
        # Recommendation based on news sentiment
        if news_data.get("status") == "success":
            sentiment = news_data.get("analysis", {}).get("sentiment", {})
            if isinstance(sentiment, dict):
                # Handle sentiment_distribution format
                if "overall_sentiment" in sentiment:
                    overall = sentiment.get("overall_sentiment", "neutral").lower()
                # Handle individual sentiment format
                else:
                    overall = sentiment.get("overall", "neutral").lower()
                
                if overall == "positive":
                    recommendations.append("News sentiment is POSITIVE - Favorable developments")
                elif overall == "negative":
                    recommendations.append("News sentiment is NEGATIVE - Monitor closely")
        
        # Recommendation based on forecast
        if forecast_data.get("status") == "success":
            forecast_analysis = forecast_data.get("analysis", {}).get("forecast_data", {})
            forecast_dir = forecast_analysis.get("forecast_direction", "neutral").lower()
            forecast_change = forecast_analysis.get("forecast_change_percent", 0)
            
            if forecast_dir == "bullish" and forecast_change > 5:
                recommendations.append(f"Forecast suggests STRONG BULLISH momentum - Potential {forecast_change:.1f}% upside in next month")
            elif forecast_dir == "bullish":
                recommendations.append("Forecast suggests BULLISH trend - Moderate upward potential expected")
            elif forecast_dir == "bearish" and forecast_change < -5:
                recommendations.append(f"Forecast suggests STRONG BEARISH momentum - Potential {abs(forecast_change):.1f}% downside risk")
            elif forecast_dir == "bearish":
                recommendations.append("Forecast suggests BEARISH trend - Downward pressure expected")
            else:
                volatility = forecast_analysis.get("volatility", 0)
                if volatility > 3:
                    recommendations.append("High volatility forecast - Consider position sizing and risk management")
        
        if not recommendations:
            recommendations.append("Gather more data for comprehensive recommendation")
        
        for i, rec in enumerate(recommendations, 1):
            report_parts.append(f"{i}. {rec}")
        
        report_parts.append(f"\n{'=' * 60}")
        
        return "\n".join(report_parts)
    
    def _generate_summary(self, combined: Dict[str, Any]) -> str:
        """Generate a summary of the combined results."""
        summary_parts = []
        summary_parts.append(f"Analysis Summary for {combined['symbol']}")
        summary_parts.append(f"Workflow Type: {combined['workflow_type']}")
        summary_parts.append(f"Nodes Executed: {', '.join(combined.get('nodes_executed', []))}")
        
        results = combined.get("results", {})
        if results:
            summary_parts.append("\nResults:")
            for key, value in results.items():
                if isinstance(value, dict):
                    status = value.get("status", "unknown")
                    summary_parts.append(f"  - {key}: {status}")
        
        if combined.get("status") == "success":
            summary_parts.append("\n Analysis completed successfully")
        elif combined.get("status") == "partial_success":
            summary_parts.append("\n Analysis completed with some errors")
        else:
            summary_parts.append("\n Analysis failed")
        
        return "\n".join(summary_parts)
    
    def run(self, symbol: str, focus: str = "comprehensive", workflow_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Execute the LangGraph workflow.
        
        Args:
            symbol: Stock symbol to analyze
            focus: Analysis focus (comprehensive, news, earnings, technical, market, forecast)
            workflow_type: Type of workflow to execute (currently only "comprehensive" - uses all specialists)
                
            Note: prompt_chaining is integrated into news specialist.
            Note: evaluator_optimizer runs automatically on combined results.
        
        Returns:
            Final state with combined results
        """
        initial_state: WorkflowState = {
            "symbol": symbol,
            "focus": focus,
            "workflow_type": "comprehensive",  # Always use comprehensive (all specialists)
            "nodes_executed": [],
            "errors": [],
            "timestamp": "",
            "news_specialist_result": {},
            "earnings_specialist_result": {},
            "market_specialist_result": {},
            "routing_result": {},
            "evaluator_optimizer_result": {},
            "combined_result": {}
        }
        
        # Invoke the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "status": "success",
            "symbol": symbol,
            "workflow_type": "comprehensive",  # Always use comprehensive (all specialists)
            "final_state": final_state,
            "result": final_state.get("combined_result", {})
        }

