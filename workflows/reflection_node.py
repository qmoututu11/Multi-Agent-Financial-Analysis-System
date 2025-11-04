"""
Reflection Node: Evaluates agent output and decides next action
Enables self-reflection and adaptive decision-making in the agentic system
"""

from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import Config


class ReflectionNode:
    """
    Reflection Node that evaluates agent outputs and decides what to do next.
    Enables agents to reflect on their work and adapt their behavior.
    """
    
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
    def reflect(self, agent_name: str, agent_output: Dict[str, Any], 
                context: Dict[str, Any], execution_plan: List[str]) -> Dict[str, Any]:
        """
        Reflect on agent output and decide next action.
        
        Args:
            agent_name: Name of the agent that just executed
            agent_output: Output from the agent
            context: Current workflow context (symbol, focus, etc.)
            execution_plan: Planned agents to run
            
        Returns:
            Dictionary with reflection results:
            - is_sufficient: Whether output is good enough
            - quality_score: Quality score (0-1)
            - gaps_identified: What information is missing
            - recommended_action: What to do next (continue, re_run, call_agent, gather_data)
            - target_agent: If calling another agent, which one
            - reasoning: Why this decision was made
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a reflection agent that evaluates the quality of specialist agent outputs
and decides what should happen next in a multi-agent financial analysis workflow.

Available agents:
- news_specialist: News sentiment and market developments
- earnings_specialist: Financial metrics and earnings
- market_specialist: Price trends and technical indicators
- forecast_specialist: Price forecasts

After an agent runs, evaluate:
1. Is the output sufficient and complete?
2. Are there gaps that need other agents?
3. Should this agent be re-run with different parameters?
4. Do we need to gather more data?

Respond in JSON format:
{{
  "is_sufficient": true/false,
  "quality_score": 0.0-1.0,
  "gaps_identified": ["gap1", "gap2"],
  "recommended_action": "continue|re_run|call_agent|gather_data",
  "target_agent": "agent_name" (if calling another agent),
  "reasoning": "Brief explanation"
}}

Actions:
- "continue": Output is good, proceed to next planned agent
- "re_run": Re-run the same agent (output was incomplete)
- "call_agent": Call a specific agent to fill gaps
- "gather_data": Need to gather more data before continuing"""),
            ("human", """Evaluate the output from {agent_name} for {symbol}.

Agent Output Status: {agent_status}
Agent Output Summary: {agent_summary}

Execution Plan: {execution_plan}
Already Executed: {executed_agents}

Reflect on whether this output is sufficient and what should happen next.""")

        ])
        
        try:
            # Extract key info from agent output
            agent_status = agent_output.get("status", "unknown")
            agent_summary = self._summarize_agent_output(agent_output)
            executed_agents = context.get("executed_agents", [])
            
            chain = prompt | self.llm
            response = chain.invoke({
                "agent_name": agent_name,
                "symbol": context.get("symbol", "STOCK"),
                "agent_status": agent_status,
                "agent_summary": agent_summary,
                "execution_plan": execution_plan,
                "executed_agents": executed_agents
            })
            
            # Parse LLM response
            import json
            try:
                reflection = json.loads(response.content.strip())
                
                # Validate action
                valid_actions = ["continue", "re_run", "call_agent", "gather_data"]
                action = reflection.get("recommended_action", "continue")
                if action not in valid_actions:
                    action = "continue"
                
                return {
                    "is_sufficient": reflection.get("is_sufficient", True),
                    "quality_score": reflection.get("quality_score", 0.7),
                    "gaps_identified": reflection.get("gaps_identified", []),
                    "recommended_action": action,
                    "target_agent": reflection.get("target_agent", ""),
                    "reasoning": reflection.get("reasoning", "Proceeding with plan"),
                    "agent_name": agent_name
                }
            except json.JSONDecodeError:
                return self._fallback_reflection(agent_status)
                
        except Exception as e:
            print(f"Reflection error: {e}, using fallback")
            return self._fallback_reflection(agent_output.get("status", "success"))
    
    def _summarize_agent_output(self, output: Dict[str, Any]) -> str:
        """Create a summary of agent output for reflection."""
        status = output.get("status", "unknown")
        
        if status == "success":
            analysis = output.get("analysis", {})
            if isinstance(analysis, dict):
                # Extract key findings
                findings = []
                if "sentiment" in analysis:
                    findings.append(f"Sentiment: {analysis.get('sentiment', {}).get('overall_sentiment', 'N/A')}")
                if "financial_metrics" in analysis:
                    findings.append("Financial metrics available")
                if "price_trends" in analysis:
                    findings.append("Price trends analyzed")
                if "forecast" in analysis:
                    findings.append("Forecast generated")
                return f"Success: {', '.join(findings) if findings else 'Analysis completed'}"
            else:
                return f"Success: Analysis completed"
        else:
            return f"Error: {output.get('error', 'Unknown error')}"
    
    def _fallback_reflection(self, status: str) -> Dict[str, Any]:
        """Fallback reflection when LLM fails."""
        if status == "success":
            return {
                "is_sufficient": True,
                "quality_score": 0.7,
                "gaps_identified": [],
                "recommended_action": "continue",
                "target_agent": "",
                "reasoning": "Fallback: Agent completed successfully, continuing",
                "agent_name": "unknown"
            }
        else:
            return {
                "is_sufficient": False,
                "quality_score": 0.3,
                "gaps_identified": ["Agent execution failed"],
                "recommended_action": "continue",  # Don't re-run on error, continue
                "target_agent": "",
                "reasoning": "Fallback: Agent failed, continuing with plan",
                "agent_name": "unknown"
            }

