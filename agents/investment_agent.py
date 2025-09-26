"""
Autonomous Investment Research Agent with Planning, Tools, Self-Reflection, and Learning
"""

from typing import Dict, Any, List
from datetime import datetime
import json
import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

from tools.langchain_tools import get_financial_tools
from config import Config

class InvestmentResearchAgent:
    """
    Autonomous Investment Research Agent with:
    - Planning research steps
    - Dynamic tool usage
    - Self-reflection
    - Learning across runs
    """
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or Config.DEFAULT_MODEL
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model_name,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
        
        # Get tools
        self.tools = get_financial_tools()
        
        # Memory for conversation
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=Config.MEMORY_WINDOW
        )
        
        # Learning storage
        self.learning_file = Config.LEARNING_FILE
        self.insights_file = Config.INSIGHTS_FILE
        self._load_learning_data()
        
        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self._get_system_prompt()),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad")
        ])
        
        # Create agent
        self.agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )
        
        # Create agent executor
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=Config.MAX_ITERATIONS
        )
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are an expert Investment Research Agent with autonomous capabilities.

Your core functions:
1. PLANNING: Plan comprehensive research steps for any stock symbol
2. TOOL USAGE: Dynamically use available tools (price data, company info, news analysis)
3. SELF-REFLECTION: Assess the quality of your own analysis
4. LEARNING: Improve future analyses based on past experiences

Available tools:
- get_stock_price: Get current stock price and market data
- get_company_info: Get company information and fundamentals
- analyze_news: Analyze news sentiment and trends

Guidelines:
- Always plan your research approach first
- Use tools to gather comprehensive data
- Provide data-driven analysis with specific metrics
- Reflect on your analysis quality
- Learn from each interaction to improve future analyses
- Be objective and cite your sources

Start each analysis by planning your research steps, then execute them systematically."""
    
    def plan_research(self, symbol: str, focus: str = "comprehensive") -> str:
        """Plan research steps for a given stock symbol."""
        planning_prompt = f"""
        Plan a comprehensive research approach for {symbol} with focus on {focus}.
        
        Consider:
        1. What data points are essential for this analysis?
        2. Which tools should be used and in what order?
        3. What key factors should be evaluated?
        4. How should the analysis be structured?
        
        Provide a clear, step-by-step research plan.
        """
        
        try:
            result = self.agent_executor.invoke({
                "input": planning_prompt,
                "chat_history": self.memory.chat_memory.messages
            })
            return result["output"]
        except Exception as e:
            return f"Planning error: {str(e)}"
    
    def research_stock(self, symbol: str, focus: str = "comprehensive") -> Dict[str, Any]:
        """Research a stock using autonomous agent capabilities."""
        print(f"\n🔍 Autonomous Investment Research: {symbol}")
        print("=" * 50)
        
        try:
            # Step 1: Plan research approach
            print(" Step 1: Planning research approach...")
            plan = self.plan_research(symbol, focus)
            print(f"Plan: {plan[:200]}...")
            
            # Step 2: Execute research
            print("\n Step 2: Executing research...")
            research_query = f"""
            Execute a comprehensive investment research analysis for {symbol}.
            
            Based on the planned approach, use the available tools to gather data and provide:
            1. Current market data and price information
            2. Company fundamentals and business overview
            3. News sentiment and market trends
            4. Investment recommendation with reasoning
            5. Risk assessment
            
            Be thorough and data-driven in your analysis.
            """
            
            result = self.agent_executor.invoke({
                "input": research_query,
                "chat_history": self.memory.chat_memory.messages
            })
            
            analysis = result["output"]
            
            # Step 3: Self-reflection
            print("\n Step 3: Self-reflection on analysis quality...")
            reflection = self.self_reflect(analysis, {"symbol": symbol, "focus": focus})
            
            # Step 4: Learning
            print("\n Step 4: Learning from this analysis...")
            self.learn_from_analysis(symbol, analysis, reflection)
            
            return {
                "symbol": symbol,
                "focus": focus,
                "plan": plan,
                "analysis": analysis,
                "reflection": reflection,
                "timestamp": datetime.now().isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            print(f"Research error: {str(e)}")
            return {
                "symbol": symbol,
                "focus": focus,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "error"
            }
    
    def self_reflect(self, analysis: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Self-reflect on the quality of the analysis."""
        reflection_prompt = f"""
        Reflect on the quality of this investment analysis for {context.get('symbol', 'stock')}:
        
        Analysis:
        {analysis}
        
        Evaluate:
        1. Completeness: Did the analysis cover all essential aspects?
        2. Accuracy: Are the data points and metrics accurate?
        3. Clarity: Is the analysis clear and well-structured?
        4. Actionability: Does it provide clear investment guidance?
        5. Areas for improvement: What could be better?
        
        Provide a brief reflection on each aspect.
        """
        
        try:
            result = self.agent_executor.invoke({
                "input": reflection_prompt,
                "chat_history": self.memory.chat_memory.messages
            })
            
            reflection = {
                "timestamp": datetime.now().isoformat(),
                "symbol": context.get("symbol", "unknown"),
                "reflection": result["output"],
                "context": context
            }
            
            return reflection
            
        except Exception as e:
            return {
                "timestamp": datetime.now().isoformat(),
                "symbol": context.get("symbol", "unknown"),
                "reflection": f"Reflection error: {str(e)}",
                "context": context
            }
    
    def learn_from_analysis(self, symbol: str, analysis: str, reflection: Dict[str, Any]) -> None:
        """Learn from the analysis and reflection."""
        # Store learning data
        learning_entry = {
            "timestamp": datetime.now().isoformat(),
            "symbol": symbol,
            "analysis_length": len(analysis),
            "reflection": reflection,
            "insights": self._extract_insights(analysis)
        }
        
        self.learning_data["analyses"].append(learning_entry)
        
        # Store symbol-specific insights
        if symbol not in self.insights_data["symbol_insights"]:
            self.insights_data["symbol_insights"][symbol] = {
                "analysis_count": 0,
                "key_insights": [],
                "successful_patterns": []
            }
        
        self.insights_data["symbol_insights"][symbol]["analysis_count"] += 1
        self.insights_data["symbol_insights"][symbol]["key_insights"].extend(
            self._extract_insights(analysis)
        )
        
        # Save learning data
        self._save_learning_data()
    
    def _extract_insights(self, analysis: str) -> List[str]:
        """Extract key insights from analysis."""
        insights = []
        
        # Simple insight extraction
        if "recommend" in analysis.lower():
            insights.append("Provided investment recommendation")
        if "risk" in analysis.lower():
            insights.append("Included risk assessment")
        if "growth" in analysis.lower():
            insights.append("Analyzed growth prospects")
        if "valuation" in analysis.lower():
            insights.append("Evaluated valuation metrics")
        
        return insights
    
    def apply_learnings(self, symbol: str) -> Dict[str, Any]:
        """Apply past learnings to improve current analysis."""
        if symbol in self.insights_data["symbol_insights"]:
            symbol_data = self.insights_data["symbol_insights"][symbol]
            return {
                "previous_analyses": symbol_data["analysis_count"],
                "key_insights": symbol_data["key_insights"][:5],  # Last 5 insights
                "successful_patterns": symbol_data["successful_patterns"]
            }
        return {"previous_analyses": 0, "key_insights": [], "successful_patterns": []}
    
    def _load_learning_data(self):
        """Load learning data from files."""
        self.learning_data = {
            "analyses": [],
            "patterns": {},
            "improvements": []
        }
        self.insights_data = {
            "symbol_insights": {},
            "market_patterns": {}
        }
        
        # Load from files if they exist
        try:
            if os.path.exists(self.learning_file):
                with open(self.learning_file, 'r') as f:
                    self.learning_data = json.load(f)
        except Exception:
            pass
        
        try:
            if os.path.exists(self.insights_file):
                with open(self.insights_file, 'r') as f:
                    self.insights_data = json.load(f)
        except Exception:
            pass
    
    def _save_learning_data(self):
        """Save learning data to files."""
        try:
            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
        except Exception:
            pass
        
        try:
            with open(self.insights_file, 'w') as f:
                json.dump(self.insights_data, f, indent=2)
        except Exception:
            pass
    
    def get_learning_summary(self) -> str:
        """Get a summary of learning insights."""
        total_analyses = len(self.learning_data["analyses"])
        symbols_analyzed = len(self.insights_data["symbol_insights"])
        
        return f"""
        Learning Summary:
        - Total analyses performed: {total_analyses}
        - Unique symbols analyzed: {symbols_analyzed}
        - Learning data stored in: {self.learning_file}
        - Insights data stored in: {self.insights_file}
        """
