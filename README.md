# Multi-Agent Financial Analysis System
## AAI-520 Group 3 Final Project

This project implements a sophisticated **Multi-Agent Financial Analysis System** using LangChain and LangGraph for autonomous investment research and analysis.

**Project Status**: ✅ **Completed**

## 🎯 Project Overview

This system demonstrates advanced AI agent architecture with:
- **Autonomous Agent Functions**: Planning, tool usage, self-reflection, and learning
- **Multi-Agent Workflow Patterns**: Prompt chaining, routing, and evaluator-optimizer
- **Real-time Financial Analysis**: Stock price, company info, and news sentiment analysis

## 🏗️ Architecture

### System Components
- **Data Sources**: Real-time financial data APIs
  - Yahoo Finance API: Stock prices, company info, news
  - SEC EDGAR API: Official regulatory filings (10-K, 10-Q, 8-K)
- **4 Specialist Agents**: LLM-powered intelligent analyzers
  - **NewsSpecialistAgent**: News sentiment analysis with prompt chaining
  - **EarningsSpecialistAgent**: Financial analysis and valuation
  - **MarketSpecialistAgent**: Technical analysis and market trends
  - **ForecastSpecialistAgent**: Historical trend analysis and price forecasting

### Agent Functions (33.8%)
- **Planning**: Autonomous research step planning
- **Tool Usage**: Dynamic API and dataset integration
- **Self-Reflection**: Quality assessment of outputs
- **Learning**: Cross-run improvement and memory

### Workflow Patterns (33.8%)
1. **Prompt Chaining**: Integrated into NewsSpecialistAgent - Ingest News → Preprocess → Classify → Extract → Summarize
2. **Comprehensive Workflow**: Coordinates all 4 specialist agents (news, earnings, market, forecast)
3. **Evaluator-Optimizer**: Automatically evaluates and optimizes combined results using LLM feedback

### Multi-Agent System Flow

```
USER REQUEST: "Analyze AAPL" (or "AAPL comprehensive")

LANGGRAPH ORCHESTRATOR:
├── ROUTES to comprehensive workflow
└── COORDINATES all specialist agents

COMPREHENSIVE WORKFLOW:
├── NewsSpecialistAgent (with Prompt Chaining)
│   ├── STEP 1: Ingest news from Yahoo Finance
│   ├── STEP 2: Preprocess with LLM
│   ├── STEP 3: Classify sentiment with LLM
│   ├── STEP 4: Extract entities with LLM
│   └── STEP 5: Summarize with LLM
│
├── EarningsSpecialistAgent
│   ├── Fetches: Company info, financial metrics from Yahoo Finance
│   ├── Fetches: SEC filings (10-K, 10-Q) from EDGAR API
│   └── LLM Analysis: Valuation assessment and financial health
│
├── MarketSpecialistAgent
│   ├── Fetches: Current price, volume, trends
│   └── LLM Analysis: Market momentum and technical insights
│
└── ForecastSpecialistAgent (NEW!)
    ├── Fetches: Historical prices (6 months)
    ├── Calculates: Trend, volatility, statistics
    └── LLM Analysis: 1-month price forecast with reasoning

EVALUATOR-OPTIMIZER (Automatic):
├── Evaluates combined analysis quality
├── Identifies weaknesses
├── Gathers additional data if needed
└── Refines analysis iteratively

FINAL OUTPUT:
├── Comprehensive financial overview
├── Market analysis
├── News sentiment summary
├── Financial forecast
└── Investment recommendations
```

### Key Distinctions
- **📊 DATA SOURCES**: Real APIs that fetch current financial data (Yahoo Finance, SEC EDGAR)
- **🤖 SPECIALIST AGENTS**: LLM-powered analyzers that interpret data and provide insights
- **🧠 LLM INTELLIGENCE**: Each agent uses LLMs for context-aware analysis, not just rule-based logic
- **🔄 WORKFLOWS**: LangGraph orchestration coordinates specialists and manages state
- **✨ AUTONOMOUS FEATURES**: Planning (LLM routing), tool usage (API calls), reflection (evaluator), learning (iterative optimization)

## 🚀 Quick Start

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd aai-520-group-3-final-project

# Install dependencies
pip install -r requirements.txt
```

### Configuration
The project uses a `.env` file for API keys. Create one with your OpenAI API key:
```bash
OPENAI_API_KEY=your-openai-api-key-here
```

**Note**: The `.env` file is already configured locally with the OpenAI API key.

### Running the System

**Option 1: Interactive Command Line**
```bash
python main.py
```

**Option 2: Jupyter Notebook Demo**
```bash
jupyter notebook demo_notebook.ipynb
```

The system provides:
- **Interactive mode** for testing individual components
- **Comprehensive analysis** using all workflows
- **Learning capabilities** that improve over time

## 📊 Demo Results

The system provides comprehensive analysis including:
- **Current market data** and price information
- **Company fundamentals** and business overview
- **News sentiment** and market trends
- **Investment recommendations** with reasoning
- **Risk assessment** and quality evaluation

## 🛠️ Technologies

- **LangChain**: Agent framework and LLM integration
- **LangGraph**: Workflow orchestration and state management
- **OpenAI GPT**: Large language model for intelligent analysis
- **Yahoo Finance API**: Real-time stock prices, company info, news
- **SEC EDGAR API**: Official regulatory filings and financial documents
- **FastAPI**: REST API backend
- **React.js**: Frontend UI (optional)
- **Python**: Core implementation language

## 📁 Project Structure

```
Multi-Agent-Financial-Analysis-System/
├── agents/
│   └── specialist_agents/       # LLM-powered specialist agents
│       ├── news_agent.py        # NewsSpecialistAgent (with prompt chaining)
│       ├── earnings_agent.py    # EarningsSpecialistAgent (with SEC filings)
│       ├── market_agent.py      # MarketSpecialistAgent (technical analysis)
│       └── forecast_agent.py   # ForecastSpecialistAgent (price forecasting)
├── workflows/
│   ├── langgraph_orchestration.py  # LangGraph workflow orchestrator
│   ├── prompt_chaining.py       # Integrated into news_agent
│   ├── routing.py               # LLM-based specialist selection
│   └── evaluator_optimizer.py   # Quality evaluation & iterative optimization
├── tools/
│   └── data_sources.py          # Yahoo Finance & SEC EDGAR API integration
├── financial-analysis-ui/       # React.js frontend (optional)
│   └── src/
│       └── AnalysisComponent.jsx
├── config.py                    # Configuration management
├── main.py                      # CLI entry point
├── api.py                       # FastAPI REST endpoint
├── .env                         # API keys
└── requirements.txt             # Dependencies
```

## 🎮 Usage Examples

### Command Line Interface
```bash
python main.py

# Interactive mode:
> AAPL                    # Comprehensive analysis (all 4 specialists)
> AAPL news               # News analysis only
> AAPL earnings           # Earnings analysis only
> AAPL market             # Market/technical analysis only
> AAPL forecast           # Forecast analysis only
```

### Python API
```python
from workflows.langgraph_orchestration import LangGraphOrchestrator

orchestrator = LangGraphOrchestrator()

# Comprehensive analysis (all specialists)
result = orchestrator.run(symbol="AAPL", focus="comprehensive")

# Focused analysis
result = orchestrator.run(symbol="AAPL", focus="forecast")
```

### REST API
```bash
# Start FastAPI server
uvicorn api:app --reload

# Analyze stock
curl -X POST "http://localhost:8000/api/analyze" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "focus": "comprehensive"}'
```

### Specialist Agents (Direct Usage)
```python
from agents.specialist_agents import (
    NewsSpecialistAgent, 
    EarningsSpecialistAgent, 
    MarketSpecialistAgent,
    ForecastSpecialistAgent
)

# News analysis with prompt chaining
news_agent = NewsSpecialistAgent(use_prompt_chaining=True)
news_result = news_agent.analyze("AAPL")

# Financial forecast
forecast_agent = ForecastSpecialistAgent()
forecast_result = forecast_agent.analyze("AAPL")
```

## 🔍 Key Features

### Autonomous Agent Functions
- ✅ **Research Planning**: Autonomous step-by-step research planning
- ✅ **Dynamic Tool Usage**: Intelligent tool selection and execution
- ✅ **Self-Reflection**: Quality assessment of own outputs
- ✅ **Learning System**: Continuous improvement from past analyses

### Multi-Agent Workflows
- ✅ **LangGraph Orchestration**: Stateful workflow management with conditional routing
- ✅ **Prompt Chaining**: Integrated LLM-powered news analysis pipeline
- ✅ **LLM-based Routing**: Intelligent specialist selection based on focus
- ✅ **Evaluator-Optimizer**: Automatic quality evaluation and iterative refinement
- ✅ **Comprehensive Workflow**: Coordinates all 4 specialist agents seamlessly

### Real-time Analysis
- ✅ **Stock Price Data**: Current prices, changes, volume from Yahoo Finance
- ✅ **Company Information**: Fundamentals, sector, industry, P/E ratios
- ✅ **SEC Filings**: Official 10-K, 10-Q, 8-K filings from EDGAR API
- ✅ **News Analysis**: LLM-powered sentiment analysis with prompt chaining
- ✅ **Market Trends**: Technical analysis and momentum indicators
- ✅ **Price Forecasting**: Historical trend analysis and 1-month forecasts
- ✅ **Investment Recommendations**: Data-driven buy/sell/hold suggestions with reasoning

## 📈 Performance Metrics

- **Analysis Quality**: Automated quality scoring (0.0-1.0)
- **Learning Progress**: Pattern recognition and improvement tracking
- **Specialist Coordination**: Multi-agent collaboration efficiency
- **Workflow Optimization**: Iterative refinement and enhancement

## 👥 Team Members

* **Maxime Boulat** - [https://github.com/MaximeBoulat](https://github.com/MaximeBoulat)
* **Qinyao Mou** - [https://github.com/qmou11](https://github.com/qmou11)
* **Dean P. Simmer** - [https://github.com/mojodean](https://github.com/mojodean)

## 📄 License

GNU GENERAL PUBLIC LICENSE Version 3

## 🎓 Course Information

This project is part of the **AAI-520** course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

---

**🎉 This multi-agent system demonstrates advanced AI architecture for autonomous financial analysis with continuous learning and improvement capabilities.**
