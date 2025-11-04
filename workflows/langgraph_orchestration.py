"""
LangGraph Orchestration Workflow
Multi-agent financial analysis with graph-based state management
"""

from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
 

from agents.specialist_agents import NewsSpecialistAgent, EarningsSpecialistAgent, MarketSpecialistAgent, ForecastSpecialistAgent
from workflows.planner_agent import PlannerAgent
from workflows.reflection_node import ReflectionNode
from config import Config


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow."""
    symbol: str
    focus: str
    workflow_type: str  # "agentic" (planner-driven, adaptive execution)
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
    
    # Agentic planning and reflection
    execution_plan: Dict[str, Any]  # Plan from planner agent
    executed_agents: Annotated[List[str], operator.add]  # Agents that have executed
    reflection_results: Annotated[List[Dict[str, Any]], operator.add]  # Reflection results
    current_iteration: int  # Current iteration number
    quality_threshold: float  # Quality threshold for stopping
    last_agent_name: str  # Last agent that executed
    last_agent_result: Dict[str, Any]  # Last agent's output


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
        
        # Initialize agentic components
        self.planner = PlannerAgent()
        self.reflector = ReflectionNode()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("planner", self._planner_node)  # NEW: Planner agent
        workflow.add_node("route", self._route_node)
        workflow.add_node("news_specialist", self._news_specialist_node)
        workflow.add_node("earnings_specialist", self._earnings_specialist_node)
        workflow.add_node("market_specialist", self._market_specialist_node)
        workflow.add_node("forecast_specialist", self._forecast_specialist_node)
        workflow.add_node("reflection", self._reflection_node)  # NEW: Reflection node
        workflow.add_node("evaluator_optimizer", self._evaluator_optimizer_node)
        workflow.add_node("combine_results", self._combine_results_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges - NEW AGENTIC FLOW
        workflow.set_entry_point("start")
        workflow.add_edge("start", "planner")  # Start with planning
        workflow.add_edge("planner", "route")  # Then route based on plan
        
        # Route to first agent based on planner's execution plan
        workflow.add_conditional_edges(
            "route",
            self._route_to_next_agent,
            {
                "news_specialist": "news_specialist",
                "earnings_specialist": "earnings_specialist",
                "market_specialist": "market_specialist",
                "forecast_specialist": "forecast_specialist",
                "combine": "combine_results"
            }
        )
        
        # After each agent, reflect on output and decide next action
        workflow.add_conditional_edges(
            "news_specialist",
            self._route_after_agent,
            {
                "reflection": "reflection",
                "combine": "combine_results"
            }
        )
        
        workflow.add_conditional_edges(
            "earnings_specialist",
            self._route_after_agent,
            {
                "reflection": "reflection",
                "combine": "combine_results"
            }
        )
        
        workflow.add_conditional_edges(
            "market_specialist",
            self._route_after_agent,
            {
                "reflection": "reflection",
                "combine": "combine_results"
            }
        )
        
        workflow.add_conditional_edges(
            "forecast_specialist",
            self._route_after_agent,
            {
                "reflection": "reflection",
                "combine": "combine_results"
            }
        )
        
        # Reflection node routes based on reflection decision
        workflow.add_conditional_edges(
            "reflection",
            self._route_after_reflection,
            {
                "continue": "route",  # Continue to next agent in plan (route back to route node)
                "combine": "combine_results"  # All done, combine results
            }
        )
        
        # After combining results, check if evaluation needed
        workflow.add_conditional_edges(
            "combine_results",
            self._should_evaluate_combined,
            {
                "evaluate": "evaluator_optimizer",
                "skip": "finalize"
            }
        )
        
        # Evaluator optimizer can iterate or finalize based on quality
        workflow.add_conditional_edges(
            "evaluator_optimizer",
            self._should_iterate_evaluation,
            {
                "iterate": "evaluator_optimizer",  # Loop back for another iteration
                "finalize": "finalize"  # Quality sufficient, finalize
            }
        )
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
            "combined_result": {},
            # Initialize agentic state
            "execution_plan": {},
            "executed_agents": [],
            "reflection_results": [],
            "current_iteration": 0,
            "quality_threshold": 0.8,
            "last_agent_name": "",
            "last_agent_result": {}
        }
    
    def _planner_node(self, state: WorkflowState) -> WorkflowState:
        """Planner agent: Creates dynamic execution plan using LLM."""
        print(f"\n Planner Agent: Creating execution plan for {state['symbol']}")
        print("=" * 60)
        
        try:
            # Create execution plan
            plan = self.planner.create_execution_plan(
                symbol=state['symbol'],
                focus=state.get('focus', 'comprehensive'),
                user_query=""  # Could be extended to accept user queries
            )
            
            print(f"  Execution Plan: {plan['reasoning']}")
            print(f"  Agents to run: {', '.join(plan['agents_to_run'])}")
            print(f"  Execution order: {', '.join(plan['execution_order'])}")
            
            return {
                **state,
                "execution_plan": plan,
                "nodes_executed": ["planner"]
            }
        except Exception as e:
            print(f"  Planner error: {e}, using fallback plan")
            # Fallback to comprehensive if planner fails
            # Note: This is just a fallback - normally the LLM decides the optimal order
            fallback_plan = {
                "agents_to_run": ["news_specialist", "earnings_specialist", "market_specialist", "forecast_specialist"],
                "execution_order": ["news_specialist", "earnings_specialist", "market_specialist", "forecast_specialist"],
                "reasoning": "Fallback: Running all agents (LLM should normally decide optimal order)",
                "estimated_iterations": 1
            }
            return {
                **state,
                "execution_plan": fallback_plan,
                "nodes_executed": ["planner"]
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
            "workflow_type": "agentic",
            "nodes_executed": ["route"]
        }
    
    def _route_to_next_agent(self, state: WorkflowState) -> str:
        """Route to next agent based on execution plan (AGENTIC)."""
        execution_plan = state.get("execution_plan", {})
        executed_agents = state.get("executed_agents", [])
        execution_order = execution_plan.get("execution_order", [])
        
        # Find next agent in plan that hasn't been executed
        for agent in execution_order:
            if agent not in executed_agents:
                print(f"  Routing to: {agent}")
                return agent
        
        # All agents in plan executed, combine results
        print("  All planned agents executed, combining results")
        return "combine"
    
    def _route_after_agent(self, state: WorkflowState) -> str:
        """Decide if reflection is needed after agent execution."""
        # Always go to reflection for agentic behavior
        return "reflection"
    
    def _route_after_reflection(self, state: WorkflowState) -> str:
        """Route based on reflection decision (AGENTIC)."""
        reflection_results = state.get("reflection_results", [])
        if not reflection_results:
            # Check if all agents in plan are done
            execution_plan = state.get("execution_plan", {})
            executed_agents = state.get("executed_agents", [])
            execution_order = execution_plan.get("execution_order", [])
            
            if len(executed_agents) >= len(execution_order):
                return "combine"  # All agents done
            else:
                return "continue"  # Continue to next agent
        
        last_reflection = reflection_results[-1]
        action = last_reflection.get("recommended_action", "continue")
        
        print(f"  Reflection decision: {action}")
        print(f"  Reasoning: {last_reflection.get('reasoning', 'N/A')}")
        
        # Check if all planned agents are done
        execution_plan = state.get("execution_plan", {})
        executed_agents = state.get("executed_agents", [])
        execution_order = execution_plan.get("execution_order", [])
        
        # If all agents done, combine regardless of reflection
        if len(executed_agents) >= len(execution_order) and action == "continue":
            print("  All planned agents executed, combining results")
            return "combine"
        
        if action == "re_run":
            # Re-run the same agent (state already updated in reflection node)
            last_agent = last_reflection.get("agent_name", "")
            if last_agent:
                print(f"  Re-running: {last_agent}")
                return "continue"  # Route back to route node
            else:
                return "continue"
        elif action == "call_agent":
            # Call specific agent (state already updated in reflection node)
            target_agent = last_reflection.get("target_agent", "")
            if target_agent:
                print(f"  Calling additional agent: {target_agent}")
                return "continue"  # Route back to route node
            else:
                return "continue"
        elif action == "gather_data":
            # Need more data - continue to next agent
            print("  Gathering more data - continuing to next agent")
            return "continue"
        else:  # "continue"
            # Check if there are more agents to run
            if len(executed_agents) < len(execution_order):
                return "continue"  # More agents to run
            else:
                return "combine"  # All done
    
    def _reflection_node(self, state: WorkflowState) -> WorkflowState:
        """Reflection node: Evaluates agent output and decides next action (AGENTIC)."""
        last_agent_name = state.get("last_agent_name", "unknown")
        last_agent_result = state.get("last_agent_result", {})
        execution_plan = state.get("execution_plan", {})
        execution_order = execution_plan.get("execution_order", [])
        
        print(f"\n Reflection Node: Evaluating {last_agent_name} output")
        print("=" * 60)
        
        try:
            # Reflect on agent output
            reflection = self.reflector.reflect(
                agent_name=last_agent_name,
                agent_output=last_agent_result,
                context={
                    "symbol": state.get("symbol", ""),
                    "focus": state.get("focus", ""),
                    "executed_agents": state.get("executed_agents", [])
                },
                execution_plan=execution_order
            )
            
            print(f"  Quality Score: {reflection['quality_score']:.2f}/1.0")
            print(f"  Is Sufficient: {reflection['is_sufficient']}")
            print(f"  Recommended Action: {reflection['recommended_action']}")
            if reflection.get('gaps_identified'):
                print(f"  Gaps Identified: {', '.join(reflection['gaps_identified'][:3])}")
            
            reflection_results = state.get("reflection_results", [])
            reflection_results = reflection_results + [reflection]
            
            # Handle state updates based on reflection decision (AGENTIC)
            updated_state = {**state}
            action = reflection.get("recommended_action", "continue")
            
            if action == "re_run":
                # Remove agent from executed list to allow re-run
                last_agent = reflection.get("agent_name", "")
                if last_agent:
                    executed_agents = updated_state.get("executed_agents", [])
                    updated_state["executed_agents"] = [a for a in executed_agents if a != last_agent]
                    print(f"  Removed {last_agent} from executed list for re-run")
            elif action == "call_agent":
                # Add target agent to execution plan
                target_agent = reflection.get("target_agent", "")
                if target_agent:
                    execution_plan = updated_state.get("execution_plan", {})
                    execution_order = execution_plan.get("execution_order", [])
                    if target_agent not in execution_order:
                        execution_order = execution_order + [target_agent]
                        execution_plan = {**execution_plan, "execution_order": execution_order}
                        updated_state["execution_plan"] = execution_plan
                        print(f"  Added {target_agent} to execution plan")
            
            updated_state["reflection_results"] = reflection_results
            updated_state["nodes_executed"] = ["reflection"]
            
            return updated_state
        except Exception as e:
            print(f"  Reflection error: {e}, using fallback")
            # Fallback: continue to next agent
            fallback_reflection = {
                "is_sufficient": True,
                "quality_score": 0.7,
                "recommended_action": "continue",
                "reasoning": "Fallback: Continuing with plan"
            }
            reflection_results = state.get("reflection_results", [])
            reflection_results = reflection_results + [fallback_reflection]
            return {
                **state,
                "reflection_results": reflection_results,
                "nodes_executed": ["reflection"]
            }
    
    def _should_evaluate_combined(self, state: WorkflowState) -> str:
        """Decide if combined routing results should be evaluated."""
        # Auto-evaluate combined results for quality improvement
        use_evaluation = state.get("use_evaluation", True)
        
        if use_evaluation:
            return "evaluate"
        else:
            return "skip"
    
    def _should_iterate_evaluation(self, state: WorkflowState) -> str:
        """Decide if evaluation should iterate (AGENTIC - iterative refinement)."""
        evaluator_result = state.get("evaluator_optimizer_result", {})
        final_evaluation = evaluator_result.get("final_evaluation", {})
        quality_score = final_evaluation.get("overall_score", 0.0)
        quality_threshold = state.get("quality_threshold", 0.8)
        current_iteration = state.get("current_iteration", 0)
        max_iterations = 3
        
        print(f"\n  Evaluation Quality Score: {quality_score:.2f}/1.0 (threshold: {quality_threshold:.2f})")
        
        if quality_score >= quality_threshold:
            print(f"  Quality sufficient, finalizing")
            return "finalize"
        elif current_iteration < max_iterations:
            print(f"  Quality below threshold, iterating (iteration {current_iteration + 1}/{max_iterations})")
            return "iterate"
        else:
            print(f"  Max iterations reached, finalizing")
            return "finalize"
    
    def _news_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the news specialist agent."""
        print(f"\n News Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.news_specialist.analyze(state['symbol'])
            # Track executed agent
            executed_agents = state.get("executed_agents", [])
            if "news_specialist" not in executed_agents:
                executed_agents = executed_agents + ["news_specialist"]
            
            return {
                **state,
                "news_specialist_result": result,
                "nodes_executed": ["news_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": result,  # For reflection
                "last_agent_name": "news_specialist"
            }
        except Exception as e:
            error_msg = f"News specialist error: {str(e)}"
            print(f" {error_msg}")
            executed_agents = state.get("executed_agents", [])
            if "news_specialist" not in executed_agents:
                executed_agents = executed_agents + ["news_specialist"]
            return {
                **state,
                "news_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["news_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": {"status": "error", "error": str(e)},
                "last_agent_name": "news_specialist"
            }
    
    def _earnings_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the earnings specialist agent."""
        print(f"\n Earnings Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.earnings_specialist.analyze(state['symbol'])
            executed_agents = state.get("executed_agents", [])
            if "earnings_specialist" not in executed_agents:
                executed_agents = executed_agents + ["earnings_specialist"]
            return {
                **state,
                "earnings_specialist_result": result,
                "nodes_executed": ["earnings_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": result,
                "last_agent_name": "earnings_specialist"
            }
        except Exception as e:
            error_msg = f"Earnings specialist error: {str(e)}"
            print(f" {error_msg}")
            executed_agents = state.get("executed_agents", [])
            if "earnings_specialist" not in executed_agents:
                executed_agents = executed_agents + ["earnings_specialist"]
            return {
                **state,
                "earnings_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["earnings_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": {"status": "error", "error": str(e)},
                "last_agent_name": "earnings_specialist"
            }
    
    def _market_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the market specialist agent."""
        print(f"\n Market Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.market_specialist.analyze(state['symbol'])
            executed_agents = state.get("executed_agents", [])
            if "market_specialist" not in executed_agents:
                executed_agents = executed_agents + ["market_specialist"]
            return {
                **state,
                "market_specialist_result": result,
                "nodes_executed": ["market_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": result,
                "last_agent_name": "market_specialist"
            }
        except Exception as e:
            error_msg = f"Market specialist error: {str(e)}"
            print(f" {error_msg}")
            executed_agents = state.get("executed_agents", [])
            if "market_specialist" not in executed_agents:
                executed_agents = executed_agents + ["market_specialist"]
            return {
                **state,
                "market_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["market_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": {"status": "error", "error": str(e)},
                "last_agent_name": "market_specialist"
            }
    
    def _forecast_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the forecast specialist agent."""
        print(f"\n Forecast Specialist: Analyzing {state['symbol']}")
        
        try:
            result = self.forecast_specialist.analyze(state['symbol'])
            executed_agents = state.get("executed_agents", [])
            if "forecast_specialist" not in executed_agents:
                executed_agents = executed_agents + ["forecast_specialist"]
            return {
                **state,
                "forecast_specialist_result": result,
                "nodes_executed": ["forecast_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": result,
                "last_agent_name": "forecast_specialist"
            }
        except Exception as e:
            error_msg = f"Forecast specialist error: {str(e)}"
            print(f" {error_msg}")
            executed_agents = state.get("executed_agents", [])
            if "forecast_specialist" not in executed_agents:
                executed_agents = executed_agents + ["forecast_specialist"]
            return {
                **state,
                "forecast_specialist_result": {"status": "error", "error": str(e)},
                "errors": [error_msg],
                "nodes_executed": ["forecast_specialist"],
                "executed_agents": executed_agents,
                "last_agent_result": {"status": "error", "error": str(e)},
                "last_agent_name": "forecast_specialist"
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
            
            # Evaluate and optimize the analysis (iterative until quality threshold)
            workflow = EvaluatorOptimizerWorkflow()
            current_iteration = state.get("current_iteration", 0)
            quality_threshold = state.get("quality_threshold", 0.8)
            
            # Use quality threshold from state for stopping condition
            result = workflow.execute_workflow(
                state['symbol'],
                state.get('focus', 'comprehensive'),
                max_iterations=3,  # Allow up to 3 iterations
                initial_analysis=analysis_text
            )
            
            # Track iteration
            final_evaluation = result.get("final_evaluation", {})
            quality_score = final_evaluation.get("overall_score", 0.0)
            
            # Update iteration count if quality is still below threshold
            if quality_score < quality_threshold and current_iteration < 2:
                state["current_iteration"] = current_iteration + 1
            
            # Update combined result with optimized version
            if state.get("combined_result"):
                state["combined_result"]["optimized_analysis"] = result.get("final_analysis", "")
                state["combined_result"]["evaluation"] = result.get("final_evaluation", {})
                # Update comprehensive analysis if optimized
                if result.get("final_analysis"):
                    state["combined_result"]["comprehensive_analysis"] = result.get("final_analysis", "")
            
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
            "workflow_type": "agentic",  # Agentic workflow with planner and reflection
            "nodes_executed": state.get('nodes_executed', []),
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Collect results from executed specialists (AGENTIC: based on actual execution, not focus)
        executed_agents = state.get("executed_agents", [])
        
        # Map agent names to result keys
        agent_result_map = {
            "news_specialist": ("news", "news_specialist_result"),
            "earnings_specialist": ("earnings", "earnings_specialist_result"),
            "market_specialist": ("market", "market_specialist_result"),
            "forecast_specialist": ("forecast", "forecast_specialist_result")
        }
        
        # Only include results from agents that actually executed
        for agent_name in executed_agents:
            if agent_name in agent_result_map:
                key, result_key = agent_result_map[agent_name]
                result = state.get(result_key, {})
                if result:  # Only add if result exists
                    combined["results"][key] = result
        
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
    
    def run(self, symbol: str, focus: str = "comprehensive", workflow_type: str = "agentic") -> Dict[str, Any]:
        """
        Execute the LangGraph agentic workflow.
        
        Uses planner agent to decide which specialists to run, and reflection nodes
        to evaluate and adapt the execution dynamically.
        
        Args:
            symbol: Stock symbol to analyze
            focus: Analysis focus (comprehensive, news, earnings, market, forecast)
            workflow_type: Type of workflow (default: "agentic" - planner-driven execution)
        
        Returns:
            Final state with combined results
        """
        initial_state: WorkflowState = {
            "symbol": symbol,
            "focus": focus,
            "workflow_type": "agentic",  # Agentic workflow with planner and reflection
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
            "workflow_type": "agentic",  # Agentic workflow with planner and reflection
            "final_state": final_state,
            "result": final_state.get("combined_result", {})
        }

