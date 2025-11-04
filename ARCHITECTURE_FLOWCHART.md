# Multi-Agent Financial Analysis System - Architecture Flowchart

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    USER REQUEST (CLI/API/Frontend)                  │
│              {symbol: "AAPL", focus: "comprehensive"}             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│              LangGraphOrchestrator (langgraph_orchestration.py)     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  START Node: Initialize State                                │  │
│  │  • Initialize nodes_executed, errors, timestamp             │  │
│  │  • Set workflow_type to "comprehensive"                       │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  ROUTE Node: Determine Specialist Execution                  │  │
│  │  • Based on focus: news, earnings, market, forecast, or all  │  │
│  │  • Routes to appropriate first specialist                     │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│         ┌───────────────┼───────────────┬──────────────┐           │
│         │               │               │              │           │
│         ▼               ▼               ▼              ▼           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐         │
│  │   NEWS    │  │ EARNINGS │  │  MARKET   │  │ FORECAST  │         │
│  │ SPECIALIST│  │SPECIALIST│  │ SPECIALIST│  │SPECIALIST │         │
│  └────┬──────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘         │
│       │               │              │             │                │
│       └───────────────┴──────────────┴─────────────┘                │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  COMBINE RESULTS Node: Merge All Specialist Results          │  │
│  │  • Collect results from executed specialists only            │  │
│  │  • Display summary of each specialist's findings             │  │
│  └──────────────────────┬───────────────────────────────────────┘  │
│                         │                                           │
│                         ▼                                           │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  EVALUATOR-OPTIMIZER Node: Quality Check & Refinement        │  │
│  │  • LLM evaluates analysis quality (0.0-1.0)                  │  │
│  │  • Identifies weaknesses and gathers additional data          │  │
│  │  • Iteratively refines analysis (up to 3 iterations)         │  │
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
**Role:** High-level orchestration and workflow coordination using LangGraph

**Responsibilities:**
- State management across the entire workflow (TypedDict with Annotated lists)
- Conditional routing based on user focus (news, earnings, market, forecast, comprehensive)
- Coordinating 4 specialist agents
- Managing execution flow (sequential with conditional skipping)
- Combining results from executed specialists
- Automatic quality evaluation and optimization

**Key Features:**
- Uses LangGraph StateGraph for workflow management
- Conditional routing based on focus parameter
- Sequential specialist execution (news → earnings → market → forecast)
- State persistence across nodes (nodes_executed, errors, results)
- Automatic evaluator-optimizer execution after combine_results

---

### 2. Specialist Agents (agents/specialist_agents/)

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

### 3. Workflows

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

## Comprehensive Workflow Example

```
User: "Analyze AAPL" (or "AAPL comprehensive")

1. START Node
   └─> Initialize state, set workflow_type="comprehensive"

2. ROUTE Node
   └─> Focus is "comprehensive" → route to news_specialist

3. NEWS SPECIALIST Node
   ├─> Fetch news from Yahoo Finance
   ├─> Prompt Chaining Workflow:
   │   ├─> Preprocess with LLM
   │   ├─> Classify sentiment with LLM
   │   ├─> Extract entities with LLM
   │   └─> Summarize with LLM
   └─> Continue to earnings_specialist (focus=comprehensive)

4. EARNINGS SPECIALIST Node
   ├─> Fetch company info from Yahoo Finance
   ├─> Fetch SEC filings (10-K, 10-Q) from EDGAR
   └─> LLM valuation analysis → Continue to market_specialist

5. MARKET SPECIALIST Node
   ├─> Fetch current price, volume
   └─> LLM trend analysis → Continue to forecast_specialist

6. FORECAST SPECIALIST Node ⭐ NEW
   ├─> Fetch historical prices (6 months)
   ├─> Calculate trend, volatility, statistics
   └─> LLM forecast generation → Continue to combine_results

7. COMBINE RESULTS Node
   ├─> Collect results from all 4 specialists
   ├─> Display summary of each specialist's findings
   └─> Continue to evaluator_optimizer

8. EVALUATOR-OPTIMIZER Node
   ├─> LLM evaluates combined analysis quality
   ├─> If score < threshold:
   │   ├─> Identify weaknesses
   │   ├─> Gather additional data if needed
   │   └─> Refine analysis (up to 3 iterations)
   └─> Continue to finalize

9. FINALIZE Node
   ├─> Generate comprehensive report:
   │   ├─> Financial Overview (earnings)
   │   ├─> Market Analysis (market)
   │   ├─> News Sentiment (news)
   │   ├─> Financial Forecast (forecast) ⭐ NEW
   │   └─> Investment Recommendations
   └─> Return final results
```

## Focus-Based Routing

```
Focus Options:
├─> "news"      → news_specialist only
├─> "earnings"  → earnings_specialist only
├─> "market"    → market_specialist only
├─> "forecast"  → market_specialist → forecast_specialist ⭐ NEW
└─> "comprehensive" → All 4 specialists sequentially
```

## Key Architecture Principles

### 1. Real Data + LLM Intelligence
- **Data Sources**: Always fetch real data from APIs (Yahoo Finance, SEC EDGAR)
- **LLM Role**: Analyze and interpret real data, not generate fake data
- **Why**: Prevents hallucination, ensures accuracy, provides current information

### 2. LLM-Powered Analysis
- All specialist agents use LLMs for intelligent analysis
- Not just rule-based thresholds - context-aware insights
- Considers sector/industry, market conditions, historical patterns

### 3. Stateful Workflow Management
- LangGraph manages state across entire workflow
- Tracks which specialists executed (nodes_executed)
- Handles errors gracefully
- Supports conditional routing based on focus

### 4. Automatic Quality Assurance
- Evaluator-optimizer automatically runs after combine_results
- LLM evaluates quality across multiple dimensions
- Iteratively refines analysis with data gathering
- Ensures high-quality output

### 5. Modular Specialist Design
- Each specialist is independent and focused
- Can run individually or as part of comprehensive workflow
- Easy to add new specialists (e.g., forecast was added)

## Comparison: Old vs New Architecture

| Aspect | Old Architecture | New Architecture |
|--------|----------------|----------------|
| **Workflow Types** | Multiple (investment_agent, routing, prompt_chaining, evaluator_optimizer) | Single comprehensive workflow |
| **Specialist Count** | 3 (news, earnings, market) | 4 (news, earnings, market, forecast) ⭐ |
| **Prompt Chaining** | Standalone workflow | Integrated into news_agent |
| **Evaluator** | Optional standalone | Automatic after combine_results |
| **Routing** | LLM-based workflow selection | Focus-based specialist routing |
| **Data Sources** | Yahoo Finance only | Yahoo Finance + SEC EDGAR ⭐ |
| **LLM Usage** | Some agents | All specialists use LLMs ⭐ |

## Technology Stack

- **LangGraph**: Workflow orchestration and state management
- **LangChain**: LLM integration and agent framework
- **OpenAI GPT**: Intelligent analysis and reasoning
- **Yahoo Finance API**: Real-time financial data
- **SEC EDGAR API**: Official regulatory filings ⭐
- **FastAPI**: REST API backend
- **React.js**: Frontend UI (optional)

## Key Takeaways

1. **LangGraph Orchestration**: Manages stateful workflow with conditional routing
2. **4 Specialist Agents**: Each uses LLM for intelligent, context-aware analysis
3. **Real Data Sources**: Yahoo Finance + SEC EDGAR prevent hallucination
4. **Automatic Quality**: Evaluator-optimizer ensures high-quality output
5. **Focus-Based Routing**: Efficient execution based on user needs
6. **Forecast Capability**: New specialist provides forward-looking predictions ⭐

---

**Last Updated**: Added ForecastSpecialistAgent and updated architecture to reflect comprehensive workflow pattern.
