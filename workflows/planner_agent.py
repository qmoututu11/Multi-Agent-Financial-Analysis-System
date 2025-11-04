"""
Hybrid Planner Agent: Combines rule-based logic with LLM for intelligent planning
More reliable than pure LLM, more flexible than pure rules
"""

from typing import Dict, Any, List, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import Config
import random


class PlannerAgent:
    """
    Hybrid Planner that combines:
    1. Rule-based decisions for common/simple cases (fast, deterministic)
    2. LLM for complex/comprehensive cases (intelligent, context-aware)
    3. Dependency graph for execution ordering (ensures logical flow)
    """
    
    # Agent dependencies: which agents should run before others
    # Format: {agent: [agents that should run before it]}
    AGENT_DEPENDENCIES = {
        "forecast_specialist": ["market_specialist", "earnings_specialist"],  # Forecast benefits from market/earnings data
        "earnings_specialist": ["market_specialist"],  # Earnings analysis benefits from current price context
        # news_specialist and market_specialist have no dependencies
    }
    
    def __init__(self, use_llm_for_comprehensive: bool = True):
        """
        Initialize hybrid planner.
        
        Args:
            use_llm_for_comprehensive: If True, use LLM for comprehensive analysis.
                                     If False, use rule-based for all cases.
        """
        self.use_llm_for_comprehensive = use_llm_for_comprehensive
        if use_llm_for_comprehensive:
            self.llm = ChatOpenAI(
                model=Config.DEFAULT_MODEL,
                temperature=0.3,  # Lower temperature for more consistent planning
                api_key=Config.OPENAI_API_KEY
            )
    
    def create_execution_plan(self, symbol: str, focus: str, user_query: str = "") -> Dict[str, Any]:
        """
        Create execution plan using hybrid approach.
        
        Strategy:
        1. Simple cases (single focus) -> Rule-based (fast, deterministic)
        2. Comprehensive cases -> LLM (intelligent, context-aware)
        3. Apply dependency ordering to ensure logical flow
        
        Args:
            symbol: Stock symbol to analyze
            focus: User-specified focus (comprehensive, news, earnings, market, forecast)
            user_query: Optional user query for additional context
            
        Returns:
            Dictionary with execution plan including:
            - agents_to_run: List of agent names to execute
            - execution_order: Suggested order (with dependencies respected)
            - reasoning: Why these agents were chosen
            - planning_method: "rule_based" or "llm_based"
        """
        focus_lower = focus.lower()
        
        # Rule-based planning for simple, focused requests
        if focus_lower in ["news", "earnings", "market", "forecast"]:
            return self._rule_based_plan(focus_lower, symbol)
        
        # LLM-based planning for comprehensive or complex cases
        if self.use_llm_for_comprehensive and focus_lower == "comprehensive":
            plan = self._llm_based_plan(symbol, focus, user_query)
            # Apply dependency ordering to LLM plan
            plan["execution_order"] = self._apply_dependency_ordering(plan["execution_order"])
            return plan
        
        # Fallback to rule-based
        return self._rule_based_plan("comprehensive", symbol)
    
    def _rule_based_plan(self, focus: str, symbol: str) -> Dict[str, Any]:
        """
        Rule-based planning for focused analysis.
        
        Rules:
        - Single focus -> primary agent + supporting agents
        - Order based on dependencies and logical flow
        """
        if focus == "news":
            agents = ["news_specialist"]
            order = ["news_specialist"]
            reasoning = "News-focused analysis: analyzing news sentiment and market developments"
        
        elif focus == "earnings":
            # Earnings analysis benefits from market context first
            agents = ["market_specialist", "earnings_specialist"]
            order = ["market_specialist", "earnings_specialist"]
            reasoning = "Earnings analysis: starting with market data for context, then financial fundamentals"
        
        elif focus == "market":
            agents = ["market_specialist"]
            order = ["market_specialist"]
            reasoning = "Market-focused analysis: analyzing price trends and technical indicators"
        
        elif focus == "forecast":
            # Forecast benefits from market and earnings data first
            agents = ["market_specialist", "earnings_specialist", "forecast_specialist"]
            order = ["market_specialist", "earnings_specialist", "forecast_specialist"]
            reasoning = "Forecast analysis: starting with market and earnings context, then generating forecast"
        
        else:  # comprehensive
            # All agents, but vary the starting point
            agents = ["news_specialist", "earnings_specialist", "market_specialist", "forecast_specialist"]
            # Randomize starting point for variety (or use dependency-based ordering)
            start_options = ["market_specialist", "earnings_specialist", "news_specialist"]
            start_agent = random.choice(start_options)
            remaining = [a for a in agents if a != start_agent]
            order = [start_agent] + remaining
            reasoning = f"Comprehensive analysis: starting with {start_agent} for context, then analyzing all aspects"
        
        return {
            "agents_to_run": agents,
            "execution_order": order,
            "reasoning": reasoning,
            "estimated_iterations": 1,
            "planning_method": "rule_based"
        }
    
    def _llm_based_plan(self, symbol: str, focus: str, user_query: str) -> Dict[str, Any]:
        """
        LLM-based planning for complex/comprehensive analysis.
        Uses simplified prompt focused on agent selection and basic ordering.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a planning agent for a financial analysis system.

Available agents:
- news_specialist: News sentiment and market developments
- earnings_specialist: Financial metrics, valuation, SEC filings
- market_specialist: Price trends, volume, technical indicators
- forecast_specialist: Price forecasts based on historical data

For comprehensive analysis, decide:
1. Which agents to run (typically all 4)
2. A logical execution order (consider: market data helps earnings analysis, earnings+market help forecast)

Note: forecast_specialist benefits from market_specialist and earnings_specialist running first.
earnings_specialist benefits from market_specialist running first.

Respond in JSON:
{{
  "agents_to_run": ["agent1", "agent2", ...],
  "execution_order": ["agent1", "agent2", ...],
  "reasoning": "Brief explanation"
}}"""),
            ("human", """Create execution plan for {symbol}.
Focus: {focus}
Query: {user_query}

Decide agents and order based on what makes sense for this analysis."""),
        ])
        
        try:
            chain = prompt | self.llm
            response = chain.invoke({
                "symbol": symbol,
                "focus": focus,
                "user_query": user_query or f"Analyze {symbol}"
            })
            
            import json
            plan = json.loads(response.content.strip())
            
            # Validate
            valid_agents = ["news_specialist", "earnings_specialist", "market_specialist", "forecast_specialist"]
            agents_to_run = [a for a in plan.get("agents_to_run", []) if a in valid_agents]
            execution_order = [a for a in plan.get("execution_order", []) if a in valid_agents]
            
            if not agents_to_run:
                # Fallback to rule-based
                return self._rule_based_plan("comprehensive", symbol)
            
            return {
                "agents_to_run": agents_to_run,
                "execution_order": execution_order,
                "reasoning": plan.get("reasoning", "LLM-generated plan"),
                "estimated_iterations": 1,
                "planning_method": "llm_based"
            }
        except Exception as e:
            print(f"LLM planning error: {e}, using rule-based fallback")
            return self._rule_based_plan("comprehensive", symbol)
    
    def _apply_dependency_ordering(self, execution_order: List[str]) -> List[str]:
        """
        Reorder execution based on agent dependencies.
        Uses topological sort to ensure dependencies run first.
        """
        if not execution_order:
            return execution_order
        
        # Build dependency graph
        dependencies = {}
        for agent in execution_order:
            dependencies[agent] = self.AGENT_DEPENDENCIES.get(agent, [])
        
        # Topological sort
        ordered = []
        remaining = set(execution_order)
        
        while remaining:
            # Find agents with no unmet dependencies
            ready = [a for a in remaining 
                    if all(dep not in remaining or dep in ordered 
                          for dep in dependencies.get(a, []))]
            
            if not ready:
                # Circular dependency or no ready agents, use original order
                return execution_order
            
            # Add ready agents (maintain relative order from original)
            for agent in execution_order:
                if agent in ready and agent not in ordered:
                    ordered.append(agent)
                    remaining.remove(agent)
        
        return ordered

