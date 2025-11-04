# Multi-Agent Financial Analysis System - Architecture Flowchart

## Agentic Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USER REQUEST (CLI/API/Frontend)                  │
│              {symbol: "AAPL", focus: "comprehensive"}             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LangGraphOrchestrator (langgraph_orchestration.py)     │
│                         AGENTIC WORKFLOW                             │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  START Node: Initialize State                                │  │
│  │  • Initialize nodes_executed, errors, timestamp             │  │
│  │  • Initialize execution_plan, executed_agents, reflection_results │
│  │  • Set workflow_type to "agentic"                             │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  PLANNER AGENT: Dynamic Execution Planning                   │  │
│  │  • LLM analyzes symbol, focus, and user query               │  │
│  │  • Decides which specialist agents to run                    │  │
│  │  • Creates execution_order (not fixed!)                     │  │
│  │  • Provides reasoning for plan                              │  │
│  │  • Example: "news_specialist → earnings_specialist"        │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ROUTE Node: Route to First Agent in Plan                    │  │
│  │  • Uses execution_plan from planner                           │  │
│  │  • Routes to first agent in execution_order                  │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│         ┌───────────────┼───────────────┬──────────────┐           │
│         │               │               │              │           │
│         ▼               ▼               ▼              ▼           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │   NEWS    │  │ EARNINGS │  │  MARKET   │  │ FORECAST  │         │
│  │ SPECIALIST│  │SPECIALIST│  │ SPECIALIST│  │SPECIALIST │         │
│  │           │  │          │  │           │  │           │         │
│  │ (if in    │  │ (if in   │  │ (if in    │  │ (if in    │         │
│  │  plan)    │  │  plan)   │  │  plan)    │  │  plan)    │         │
│  └────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │               │              │             │                │
│       └───────────────┴──────────────┴─────────────┘                │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  REFLECTION Node: Evaluate Agent Output                       │  │
│  │  • LLM evaluates output quality (0.0-1.0)                   │  │
│  │  • Identifies gaps and weaknesses                            │  │
│  │  • Decides next action:                                      │  │
│  │    - continue: Output good, proceed to next agent            │  │
│  │    - re_run: Output incomplete, re-run current agent         │  │
│  │    - call_agent: Need additional agent to fill gaps          │  │
│  │    - gather_data: Need more information                       │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│         ┌───────────────┼───────────────┬──────────────┐           │
│         │               │               │              │           │
│         ▼               ▼               ▼              ▼           │
│    CONTINUE        RE_RUN          CALL_AGENT    GATHER_DATA      │
│    (to next)      (same agent)    (new agent)    (more data)      │
│         │               │               │              │           │
│         └───────────────┴───────────────┴──────────────┘           │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  COMBINE RESULTS Node: Merge All Executed Agents            │  │
│  │  • Collect results from executed_agents (not fixed list)    │  │
│  │  • Display summary of each specialist's findings             │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  EVALUATOR-OPTIMIZER Node: Quality Check & Refinement        │  │
│  │  • LLM evaluates combined analysis quality (0.0-1.0)         │  │
│  │  • Identifies weaknesses and gathers additional data          │  │
│  │  • Iteratively refines analysis (up to 3 iterations)          │  │
│  │  • Continues until quality threshold met                      │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  FINALIZE Node: Generate Comprehensive Report                 │  │
│  │  • Financial overview                                          │  │
│  │  • Market analysis                                            │  │
│  │  • News sentiment                                             │  │
│  │  • Financial forecast                                         │  │
│  │  • Investment recommendations                                 │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
└─────────────────────────┼──────────────────────────────────────────┘
                          │
                          ▼
                 ┌────────────────┐
                 │  Final Results  │
                 │  (JSON/Report)  │
                 └────────────────┘
```

## Detailed Component Breakdown

### 1. LangGraphOrchestrator (langgraph_orchestration.py)
**Role:** Agentic workflow orchestration with dynamic planning and reflection

**Responsibilities:**
- State management across the entire workflow (TypedDict with Annotated lists)
- **Planner Agent Integration**: LLM decides which specialists to run
- **Reflection Node Integration**: Evaluates each agent's output and adapts execution
- Dynamic routing based on execution plan and reflection decisions
- Coordinating 4 specialist agents (only those in the plan)
- Managing execution flow (adaptive, not fixed)
- Combining results from executed specialists
- Automatic quality evaluation and optimization

**Key Features:**
- Uses LangGraph StateGraph for workflow management
- **Agentic Planning**: Planner Agent creates dynamic execution plan
- **Adaptive Execution**: Reflection Nodes evaluate and adapt after each agent
- State persistence across nodes (nodes_executed, executed_agents, reflection_results)
- Automatic evaluator-optimizer execution after combine_results

---

### 2. Planner Agent (workflows/planner_agent.py) ⭐ NEW
**Role:** LLM-powered dynamic execution planning

**Responsibilities:**
- Analyzes user query, symbol, and focus
- Decides which specialist agents should run
- Creates execution order (not necessarily all agents)
- Provides reasoning for the plan

**LLM Prompt:**
- Describes available specialist agents
- Asks LLM to decide which agents are needed
- Requests execution order and reasoning

**Output:**
```json
{
  "agents_to_run": ["news_specialist", "earnings_specialist"],
  "execution_order": ["news_specialist", "earnings_specialist"],
  "reasoning": "User wants comprehensive analysis, so all agents needed",
  "estimated_iterations": 1
}
```

**Fallback:** If LLM fails, defaults to all 4 agents for "comprehensive" focus

---

### 3. Reflection Node (workflows/reflection_node.py) ⭐ NEW
**Role:** Self-evaluation and adaptive decision-making

**Responsibilities:**
- Evaluates agent output quality after each specialist runs
- Identifies gaps and weaknesses in the output
- Decides next action based on evaluation

**LLM Evaluation:**
- Quality score (0.0-1.0)
- Gaps identified
- Recommended action

**Actions:**
- **continue**: Output is sufficient, proceed to next agent in plan
- **re_run**: Output incomplete, remove agent from executed_agents and re-run
- **call_agent**: Add new agent to execution_plan to fill gaps
- **gather_data**: Need more information before proceeding

**Output:**
```json
{
  "is_sufficient": true/false,
  "quality_score": 0.85,
  "gaps_identified": ["missing price data"],
  "recommended_action": "gather_data",
  "target_agent": null,
  "reasoning": "Output is good but missing current price"
}
```

---

### 4. Specialist Agents (agents/specialist_agents/)

#### NewsSpecialistAgent (news_agent.py)
**Role:** News sentiment analysis with LLM-powered prompt chaining

**Responsibilities:**
- Fetch recent news articles from Yahoo Finance
- Use PromptChainingWorkflow for enhanced analysis:
  - Ingest news articles
  - Preprocess with LLM
  - Classify sentiment with LLM
  - Extract entities with LLM
  - Summarize with LLM
- Provide sentiment distribution (positive/negative/neutral)
- Generate LLM summary of news

**Data Sources:**
- Yahoo Finance API: News articles

**LLM Usage:**
- Sentiment classification
- Entity extraction
- News summarization

---

#### EarningsSpecialistAgent (earnings_agent.py)
**Role:** Financial analysis and valuation assessment

**Responsibilities:**
- Fetch company fundamentals (P/E ratio, market cap, sector, industry)
- Fetch SEC filings (10-K, 10-Q) from EDGAR API
- LLM-powered valuation assessment:
  - Context-aware analysis (considers sector/industry)
  - Financial health insights
  - Valuation assessment (undervalued/fairly valued/overvalued)

**Data Sources:**
- Yahoo Finance API: Company info, financial metrics
- SEC EDGAR API: Official regulatory filings

**LLM Usage:**
- Valuation assessment (not just rule-based thresholds)
- Financial health analysis
- Sector/industry context consideration

---

#### MarketSpecialistAgent (market_agent.py)
**Role:** Technical analysis and market trend analysis

**Responsibilities:**
- Fetch current price, volume, price changes
- LLM-powered trend analysis:
  - Market momentum interpretation
  - Volume signal analysis
  - Technical insights

**Data Sources:**
- Yahoo Finance API: Stock price, volume

**LLM Usage:**
- Trend interpretation (bullish/bearish/neutral)
- Volume signal analysis
- Technical insights

---

#### ForecastSpecialistAgent (forecast_agent.py)
**Role:** Historical trend analysis and price forecasting

**Responsibilities:**
- Fetch historical price data (6 months)
- Calculate statistics:
  - Trend direction (uptrend/downtrend/sideways)
  - Volatility (standard deviation of returns)
  - Price ranges (min, max, average)
- LLM-powered forecasting:
  - Analyze historical patterns
  - Generate 1-month price forecast
  - Provide forecast direction (bullish/bearish/neutral)
  - Explain forecast reasoning

**Data Sources:**
- Yahoo Finance API: Historical prices (6mo, 1y, 2y, 5y, max)

**LLM Usage:**
- Pattern recognition from historical data
- Forward-looking price predictions
- Forecast confidence assessment
- Risk analysis based on volatility

---

### 5. Workflows

#### PromptChainingWorkflow (prompt_chaining.py)
**Role:** Sequential LLM-powered news processing pipeline

**Status:** Integrated into NewsSpecialistAgent (not standalone)

**Steps:**
1. **Ingest**: Fetch news articles from Yahoo Finance
2. **Preprocess**: LLM cleans and structures text
3. **Classify**: LLM determines sentiment (positive/negative/neutral)
4. **Extract**: LLM identifies key entities and topics
5. **Summarize**: LLM generates comprehensive summary

---

#### EvaluatorOptimizerWorkflow (evaluator_optimizer.py)
**Role:** Automatic quality evaluation and iterative refinement

**Status:** Automatically runs after combine_results node

**Process:**
1. **Evaluate**: LLM evaluates analysis quality (0.0-1.0)
   - Dimensions: Completeness, Accuracy, Clarity, Actionability, Depth
   - Provides strengths, weaknesses, recommendations
2. **Optimize**: LLM refines analysis based on feedback
   - Addresses identified weaknesses
   - Gathers additional data if needed (price, company info, news)
   - Iterates up to 3 times until quality threshold met
3. **Final Evaluation**: Validates improved quality

**Key Feature:** Dynamic data gathering during iterations based on LLM-identified needs

---

## Agentic Workflow Example

```
User: "Analyze AAPL" (or "AAPL comprehensive")

1. START Node
   └─> Initialize state, set workflow_type="agentic"
        Initialize execution_plan={}, executed_agents=[]

2. PLANNER AGENT Node ⭐
   ├─> LLM analyzes: symbol="AAPL", focus="comprehensive"
   ├─> LLM decides: "User wants comprehensive analysis, need all 4 agents"
   └─> Creates plan:
       {
         "agents_to_run": ["news_specialist", "earnings_specialist", 
                          "market_specialist", "forecast_specialist"],
         "execution_order": ["news_specialist", "earnings_specialist", 
                            "market_specialist", "forecast_specialist"],
         "reasoning": "Comprehensive focus requires all specialists"
       }

3. ROUTE Node
   └─> Routes to first agent in plan: news_specialist

4. NEWS SPECIALIST Node
   ├─> Fetch news from Yahoo Finance
   ├─> Prompt Chaining Workflow:
   │   ├─> Preprocess with LLM
   │   ├─> Classify sentiment with LLM
   │   ├─> Extract entities with LLM
   │   └─> Summarize with LLM
   ├─> Update executed_agents: ["news_specialist"]
   └─> → REFLECTION Node

5. REFLECTION Node ⭐
   ├─> LLM evaluates news_specialist output:
   │   - Quality score: 0.85
   │   - Gaps: None
   │   - Action: continue
   └─> Decision: continue to next agent

6. ROUTE Node (from reflection)
   └─> Routes to next agent in plan: earnings_specialist

7. EARNINGS SPECIALIST Node
   ├─> Fetch company info from Yahoo Finance
   ├─> Fetch SEC filings (10-K, 10-Q) from EDGAR
   ├─> LLM valuation analysis
   ├─> Update executed_agents: ["news_specialist", "earnings_specialist"]
   └─> → REFLECTION Node

8. REFLECTION Node ⭐
   ├─> LLM evaluates earnings_specialist output:
   │   - Quality score: 0.70
   │   - Gaps: ["missing current price data"]
   │   - Action: gather_data
   └─> Decision: gather more data (price) before continuing

9. ROUTE Node (from reflection - gather_data)
   └─> Routes to market_specialist (to get price data)

10. MARKET SPECIALIST Node
    ├─> Fetch current price, volume
    ├─> LLM trend analysis
    ├─> Update executed_agents: ["news_specialist", "earnings_specialist", "market_specialist"]
    └─> → REFLECTION Node

11. REFLECTION Node ⭐
    ├─> LLM evaluates market_specialist output:
    │    - Quality score: 0.90
    │    - Gaps: None
    │    - Action: continue
    └─> Decision: continue to next agent in plan

12. ROUTE Node (from reflection)
    └─> Routes to next agent in plan: forecast_specialist

13. FORECAST SPECIALIST Node
    ├─> Fetch historical prices (6 months)
    ├─> Calculate trend, volatility, statistics
    ├─> LLM forecast generation
    ├─> Update executed_agents: ["news_specialist", "earnings_specialist", 
                                 "market_specialist", "forecast_specialist"]
    └─> → REFLECTION Node

14. REFLECTION Node ⭐
    ├─> LLM evaluates forecast_specialist output:
    │    - Quality score: 0.88
    │    - Gaps: None
    │    - Action: continue
    └─> Decision: all agents in plan executed, proceed to combine_results

15. COMBINE RESULTS Node
    ├─> Collect results from executed_agents:
    │    - news_specialist_result
    │    - earnings_specialist_result
    │    - market_specialist_result
    │    - forecast_specialist_result
    ├─> Display summary of each specialist's findings
    └─> Continue to evaluator_optimizer

16. EVALUATOR-OPTIMIZER Node
    ├─> LLM evaluates combined analysis quality
    ├─> If score < threshold:
    │   ├─> Identify weaknesses
    │   ├─> Gather additional data if needed
    │   └─> Refine analysis (up to 3 iterations)
    └─> Continue to finalize

17. FINALIZE Node
    ├─> Generate comprehensive report:
    │   ├─> Financial Overview (earnings)
    │   ├─> Market Analysis (market)
    │   ├─> News Sentiment (news)
    │   ├─> Financial Forecast (forecast)
    │   └─> Investment Recommendations
    └─> Return final results
```

## Focus-Based Planning

```
Focus Options (affects Planner Agent's decision):
├─> "news"      → Planner may decide: news_specialist only
├─> "earnings"  → Planner may decide: earnings_specialist + market_specialist (for context)
├─> "market"    → Planner may decide: market_specialist + forecast_specialist
├─> "forecast"  → Planner may decide: forecast_specialist + market_specialist (for context)
└─> "comprehensive" → Planner decides: all 4 specialists

Note: Planner Agent makes intelligent decisions, not hard-coded rules!
```

## Key Architecture Principles

### 1. Agentic Planning (Not Fixed Flow)
- **Planner Agent**: LLM decides which agents to run based on query
- **Dynamic Execution**: Execution order adapts based on plan, not fixed sequence
- **Intelligent Selection**: Planner considers context, focus, and user needs

### 2. Self-Reflection and Adaptation
- **Reflection Nodes**: Each agent's output is evaluated by LLM
- **Adaptive Routing**: Execution adapts based on reflection decisions
- **Quality-Driven**: System continues until quality threshold met

### 3. Real Data + LLM Intelligence
- **Data Sources**: Always fetch real data from APIs (Yahoo Finance, SEC EDGAR)
- **LLM Role**: Analyze and interpret real data, not generate fake data
- **Why**: Prevents hallucination, ensures accuracy, provides current information

### 4. LLM-Powered Analysis
- All specialist agents use LLMs for intelligent analysis
- Not just rule-based thresholds - context-aware insights
- Considers sector/industry, market conditions, historical patterns

### 5. Stateful Workflow Management
- LangGraph manages state across entire workflow
- Tracks which specialists executed (executed_agents)
- Tracks execution plan (execution_plan)
- Tracks reflection results (reflection_results)
- Handles errors gracefully

### 6. Automatic Quality Assurance
- Evaluator-optimizer automatically runs after combine_results
- LLM evaluates quality across multiple dimensions
- Iteratively refines analysis with data gathering
- Ensures high-quality output

### 7. Modular Specialist Design
- Each specialist is independent and focused
- Can run individually or as part of agentic workflow
- Easy to add new specialists
- Planner decides which specialists to use

## Comparison: Old vs New Architecture

| Aspect | Old Architecture | New Architecture (Agentic) |
|--------|----------------|----------------|
| **Workflow Type** | "comprehensive" (fixed) | "agentic" (planner-driven) |
| **Execution Flow** | Fixed sequence (all agents) | Dynamic (planner decides) |
| **Routing Logic** | Focus-based (hard-coded) | Planner-based (LLM-decided) |
| **Quality Control** | Only at end (evaluator) | After each agent (reflection) |
| **Adaptation** | None | Reflection nodes adapt execution |
| **Specialist Count** | 4 (news, earnings, market, forecast) | 4 (same, but planner decides usage) |
| **Prompt Chaining** | Integrated into news_agent | Same |
| **Evaluator** | Automatic after combine | Same |
| **Data Sources** | Yahoo Finance + SEC EDGAR | Same |
| **LLM Usage** | All specialists use LLMs | Same + Planner + Reflection |

## Technology Stack

- **LangGraph**: Agentic workflow orchestration and state management
- **LangChain**: LLM integration and agent framework
- **OpenAI GPT**: Intelligent analysis, planning, and reflection
- **Yahoo Finance API**: Real-time financial data
- **SEC EDGAR API**: Official regulatory filings
- **FastAPI**: REST API backend
- **React.js**: Frontend UI (optional)

## Key Takeaways

1. **Agentic Architecture**: Planner Agent decides which agents to run, Reflection Nodes evaluate output
2. **Dynamic Execution**: Execution adapts based on LLM decisions, not fixed flow
3. **Self-Reflection**: Each agent's output is evaluated and execution adapts accordingly
4. **4 Specialist Agents**: Each uses LLM for intelligent, context-aware analysis
5. **Real Data Sources**: Yahoo Finance + SEC EDGAR prevent hallucination
6. **Automatic Quality**: Evaluator-optimizer ensures high-quality output
7. **Adaptive Routing**: System adapts based on reflection decisions

---

**Last Updated**: Implemented agentic architecture with Planner Agent and Reflection Nodes for dynamic, adaptive execution.