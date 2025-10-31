"""
LangGraph Orchestration Workflow
Multi-agent financial analysis with graph-based state management
"""

from typing import Dict, Any, List, TypedDict, Annotated
from datetime import datetime
import operator

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
 

from agents.investment_agent import InvestmentResearchAgent
from agents.specialist_agents import NewsSpecialistAgent, EarningsSpecialistAgent, MarketSpecialistAgent
from config import Config


class WorkflowState(TypedDict):
    """State schema for the LangGraph workflow."""
    symbol: str
    focus: str
    workflow_type: str  # "agent", "routing", "prompt_chaining", "evaluator_optimizer", "comprehensive"
    
    # Agent outputs
    investment_agent_result: Dict[str, Any]
    news_specialist_result: Dict[str, Any]
    earnings_specialist_result: Dict[str, Any]
    market_specialist_result: Dict[str, Any]
    
    # Workflow results
    routing_result: Dict[str, Any]
    prompt_chaining_result: Dict[str, Any]
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
        self.investment_agent = InvestmentResearchAgent()
        self.news_specialist = NewsSpecialistAgent()
        self.earnings_specialist = EarningsSpecialistAgent()
        self.market_specialist = MarketSpecialistAgent()
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow graph."""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("route", self._route_node)
        workflow.add_node("investment_agent", self._investment_agent_node)
        workflow.add_node("news_specialist", self._news_specialist_node)
        workflow.add_node("earnings_specialist", self._earnings_specialist_node)
        workflow.add_node("market_specialist", self._market_specialist_node)
        workflow.add_node("prompt_chaining", self._prompt_chaining_node)
        workflow.add_node("evaluator_optimizer", self._evaluator_optimizer_node)
        workflow.add_node("combine_results", self._combine_results_node)
        workflow.add_node("finalize", self._finalize_node)
        
        # Define edges
        workflow.set_entry_point("start")
        workflow.add_edge("start", "route")
        workflow.add_conditional_edges(
            "route",
            self._route_decision,
            {
                "investment_agent": "investment_agent",
                "routing": "news_specialist",
                "prompt_chaining": "prompt_chaining",
                "evaluator_optimizer": "evaluator_optimizer",
                "comprehensive": "news_specialist"
            }
        )
        
        # Routing workflow edges (parallel execution of specialists)
        workflow.add_edge("news_specialist", "earnings_specialist")
        workflow.add_edge("earnings_specialist", "market_specialist")
        workflow.add_edge("market_specialist", "combine_results")
        
        # Direct paths
        workflow.add_edge("investment_agent", "combine_results")
        workflow.add_edge("prompt_chaining", "combine_results")
        workflow.add_edge("evaluator_optimizer", "combine_results")
        
        # Finalize
        workflow.add_edge("combine_results", "finalize")
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    def _start_node(self, state: WorkflowState) -> WorkflowState:
        """Initialize the workflow state."""
        print(f"\n🚀 LangGraph Orchestration: Starting analysis for {state['symbol']}")
        print("=" * 60)
        
        return {
            **state,
            "nodes_executed": ["start"],
            "errors": [],
            "timestamp": datetime.now().isoformat(),
            "investment_agent_result": {},
            "news_specialist_result": {},
            "earnings_specialist_result": {},
            "market_specialist_result": {},
            "routing_result": {},
            "prompt_chaining_result": {},
            "evaluator_optimizer_result": {},
            "combined_result": {}
        }
    
    def _route_node(self, state: WorkflowState) -> WorkflowState:
        """Route to appropriate workflow based on focus or workflow_type."""
        print(f"\n📍 Routing: workflow_type={state.get('workflow_type', 'comprehensive')}")
        
        # If workflow_type is explicitly set, use it
        workflow_type = state.get('workflow_type', 'comprehensive')
        
        # If workflow_type is comprehensive but focus suggests routing, adjust
        if workflow_type == 'comprehensive':
            focus = state.get('focus', 'comprehensive').lower()
            if focus in ['news', 'earnings', 'technical', 'market']:
                workflow_type = 'routing'
        
        return {
            **state,
            "workflow_type": workflow_type,
            "nodes_executed": state.get("nodes_executed", []) + ["route"]
        }
    
    def _route_decision(self, state: WorkflowState) -> str:
        """Determine which path to take based on workflow_type."""
        workflow_type = state.get('workflow_type', 'comprehensive')
        
        if workflow_type == 'routing':
            return 'routing'  # Will execute news -> earnings -> market -> combine
        elif workflow_type == 'comprehensive':
            return 'comprehensive'  # Will execute all specialists
        elif workflow_type == 'investment_agent':
            return 'investment_agent'
        elif workflow_type == 'prompt_chaining':
            return 'prompt_chaining'
        elif workflow_type == 'evaluator_optimizer':
            return 'evaluator_optimizer'
        else:
            return 'comprehensive'  # Default to comprehensive
    
    def _investment_agent_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the investment research agent."""
        print(f"\n🤖 Investment Agent Node: Analyzing {state['symbol']}")
        
        try:
            result = self.investment_agent.research_stock(
                state['symbol'],
                state.get('focus', 'comprehensive')
            )
            return {
                **state,
                "investment_agent_result": result,
                "nodes_executed": state.get("nodes_executed", []) + ["investment_agent"]
            }
        except Exception as e:
            error_msg = f"Investment agent error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "investment_agent_result": {"status": "error", "error": str(e)},
                "errors": state.get("errors", []) + [error_msg],
                "nodes_executed": state.get("nodes_executed", []) + ["investment_agent"]
            }
    
    def _news_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the news specialist agent."""
        print(f"\n📰 News Specialist Node: Analyzing {state['symbol']}")
        
        try:
            result = self.news_specialist.analyze(state['symbol'])
            return {
                **state,
                "news_specialist_result": result,
                "nodes_executed": state.get("nodes_executed", []) + ["news_specialist"]
            }
        except Exception as e:
            error_msg = f"News specialist error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "news_specialist_result": {"status": "error", "error": str(e)},
                "errors": state.get("errors", []) + [error_msg],
                "nodes_executed": state.get("nodes_executed", []) + ["news_specialist"]
            }
    
    def _earnings_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the earnings specialist agent."""
        print(f"\n💰 Earnings Specialist Node: Analyzing {state['symbol']}")
        
        try:
            result = self.earnings_specialist.analyze(state['symbol'])
            return {
                **state,
                "earnings_specialist_result": result,
                "nodes_executed": state.get("nodes_executed", []) + ["earnings_specialist"]
            }
        except Exception as e:
            error_msg = f"Earnings specialist error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "earnings_specialist_result": {"status": "error", "error": str(e)},
                "errors": state.get("errors", []) + [error_msg],
                "nodes_executed": state.get("nodes_executed", []) + ["earnings_specialist"]
            }
    
    def _market_specialist_node(self, state: WorkflowState) -> WorkflowState:
        """Execute the market specialist agent."""
        print(f"\n📈 Market Specialist Node: Analyzing {state['symbol']}")
        
        try:
            result = self.market_specialist.analyze(state['symbol'])
            
            # Store routing results if this is part of routing workflow
            if state.get('workflow_type') in ['routing', 'comprehensive']:
                routing_result = {
                    "symbol": state['symbol'],
                    "specialists_used": ["news", "earnings", "market"],
                    "analyses": {
                        "news": state.get("news_specialist_result", {}),
                        "earnings": state.get("earnings_specialist_result", {}),
                        "market": result
                    },
                    "status": "success",
                    "timestamp": datetime.now().isoformat()
                }
                return {
                    **state,
                    "market_specialist_result": result,
                    "routing_result": routing_result,
                    "nodes_executed": state.get("nodes_executed", []) + ["market_specialist"]
                }
            
            return {
                **state,
                "market_specialist_result": result,
                "nodes_executed": state.get("nodes_executed", []) + ["market_specialist"]
            }
        except Exception as e:
            error_msg = f"Market specialist error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "market_specialist_result": {"status": "error", "error": str(e)},
                "errors": state.get("errors", []) + [error_msg],
                "nodes_executed": state.get("nodes_executed", []) + ["market_specialist"]
            }
    
    def _prompt_chaining_node(self, state: WorkflowState) -> WorkflowState:
        """Execute prompt chaining workflow."""
        print(f"\n🔗 Prompt Chaining Node: Processing {state['symbol']}")
        
        try:
            from workflows.prompt_chaining import PromptChainingWorkflow
            workflow = PromptChainingWorkflow()
            result = workflow.execute_workflow(state['symbol'], max_articles=5)
            
            return {
                **state,
                "prompt_chaining_result": result,
                "nodes_executed": state.get("nodes_executed", []) + ["prompt_chaining"]
            }
        except Exception as e:
            error_msg = f"Prompt chaining error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "prompt_chaining_result": {"status": "error", "error": str(e)},
                "errors": state.get("errors", []) + [error_msg],
                "nodes_executed": state.get("nodes_executed", []) + ["prompt_chaining"]
            }
    
    def _evaluator_optimizer_node(self, state: WorkflowState) -> WorkflowState:
        """Execute evaluator-optimizer workflow."""
        print(f"\n⚖️ Evaluator-Optimizer Node: Processing {state['symbol']}")
        
        try:
            from workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
            workflow = EvaluatorOptimizerWorkflow()
            result = workflow.execute_workflow(
                state['symbol'],
                state.get('focus', 'comprehensive'),
                max_iterations=2
            )
            
            return {
                **state,
                "evaluator_optimizer_result": result,
                "nodes_executed": state.get("nodes_executed", []) + ["evaluator_optimizer"]
            }
        except Exception as e:
            error_msg = f"Evaluator-optimizer error: {str(e)}"
            print(f"❌ {error_msg}")
            return {
                **state,
                "evaluator_optimizer_result": {"status": "error", "error": str(e)},
                "errors": state.get("errors", []) + [error_msg],
                "nodes_executed": state.get("nodes_executed", []) + ["evaluator_optimizer"]
            }
    
    def _combine_results_node(self, state: WorkflowState) -> WorkflowState:
        """Combine results from all executed nodes."""
        print(f"\n🔗 Combining Results")
        
        workflow_type = state.get('workflow_type', 'comprehensive')
        combined = {
            "symbol": state['symbol'],
            "focus": state.get('focus', 'comprehensive'),
            "workflow_type": workflow_type,
            "nodes_executed": state.get('nodes_executed', []),
            "timestamp": datetime.now().isoformat(),
            "results": {}
        }
        
        # Collect results based on workflow type
        if workflow_type == 'investment_agent':
            combined["results"]["investment_agent"] = state.get("investment_agent_result", {})
        
        elif workflow_type in ['routing', 'comprehensive']:
            combined["results"]["routing"] = state.get("routing_result", {})
            combined["results"]["news"] = state.get("news_specialist_result", {})
            combined["results"]["earnings"] = state.get("earnings_specialist_result", {})
            combined["results"]["market"] = state.get("market_specialist_result", {})
        
        elif workflow_type == 'prompt_chaining':
            combined["results"]["prompt_chaining"] = state.get("prompt_chaining_result", {})
        
        elif workflow_type == 'evaluator_optimizer':
            combined["results"]["evaluator_optimizer"] = state.get("evaluator_optimizer_result", {})
        
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
            "nodes_executed": state.get("nodes_executed", []) + ["combine_results"]
        }
    
    def _finalize_node(self, state: WorkflowState) -> WorkflowState:
        """Finalize the workflow."""
        print(f"\n✅ Finalizing Results")
        print("=" * 60)
        
        return {
            **state,
            "nodes_executed": state.get("nodes_executed", []) + ["finalize"]
        }
    
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
            summary_parts.append("\n✅ Analysis completed successfully")
        elif combined.get("status") == "partial_success":
            summary_parts.append("\n⚠️ Analysis completed with some errors")
        else:
            summary_parts.append("\n❌ Analysis failed")
        
        return "\n".join(summary_parts)
    
    def run(self, symbol: str, focus: str = "comprehensive", workflow_type: str = "comprehensive") -> Dict[str, Any]:
        """
        Execute the LangGraph workflow.
        
        Args:
            symbol: Stock symbol to analyze
            focus: Analysis focus (comprehensive, news, earnings, technical, market)
            workflow_type: Type of workflow to execute
                - "investment_agent": Use main investment agent
                - "routing": Route to specialist agents
                - "prompt_chaining": Execute prompt chaining workflow
                - "evaluator_optimizer": Execute evaluator-optimizer workflow
                - "comprehensive": Execute all specialists (routing)
        
        Returns:
            Final state with combined results
        """
        initial_state: WorkflowState = {
            "symbol": symbol,
            "focus": focus,
            "workflow_type": workflow_type,
            "nodes_executed": [],
            "errors": [],
            "timestamp": "",
            "investment_agent_result": {},
            "news_specialist_result": {},
            "earnings_specialist_result": {},
            "market_specialist_result": {},
            "routing_result": {},
            "prompt_chaining_result": {},
            "evaluator_optimizer_result": {},
            "combined_result": {}
        }
        
        # Invoke the graph
        final_state = self.graph.invoke(initial_state)
        
        return {
            "status": "success",
            "symbol": symbol,
            "workflow_type": workflow_type,
            "final_state": final_state,
            "result": final_state.get("combined_result", {})
        }

