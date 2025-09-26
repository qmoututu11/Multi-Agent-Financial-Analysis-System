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
- **5 Tools**: Simple data fetchers and analyzers
- **4 Agents**: Intelligent decision-makers that use the tools
  - **1 Main Agent**: InvestmentResearchAgent (orchestrates everything and learns)
  - **3 Specialist Agents**: NewsSpecialistAgent, EarningsSpecialistAgent, MarketSpecialistAgent (in agents/specialist_agents/)

### Agent Functions (33.8%)
- **Planning**: Autonomous research step planning
- **Tool Usage**: Dynamic API and dataset integration
- **Self-Reflection**: Quality assessment of outputs
- **Learning**: Cross-run improvement and memory

### Workflow Patterns (33.8%)
1. **Prompt Chaining**: Ingest News → Preprocess → Classify → Extract → Summarize
2. **Routing**: Direct content to specialist agents (earnings, news, market analyzers)
3. **Evaluator-Optimizer**: Generate → Evaluate → Refine using feedback

### Multi-Agent System Flow

```
USER REQUEST: "Analyze AAPL"

MAIN AGENT (InvestmentResearchAgent):
├── PLANNING: "I need comprehensive analysis"
├── TOOL USAGE: Uses 5 tools (price, company, news, financial, technical)
├── SELF-REFLECTION: "My analysis was good but could improve"
└── LEARNING: "I'll remember this approach worked well"

ROUTING WORKFLOW:
├── ROUTES to: NewsSpecialist + EarningsSpecialist + MarketSpecialist
├── Each specialist uses their preferred tools
└── COMBINES results into comprehensive analysis

PROMPT CHAINING WORKFLOW:
├── STEP 1: Ingest news (NewsAnalysisTool)
├── STEP 2: Preprocess text (TextPreprocessor)
├── STEP 3: Classify sentiment (SentimentClassifier)
├── STEP 4: Extract entities (EntityExtractor)
└── STEP 5: Summarize (TextSummarizer)
```

### Key Distinctions
- **🛠️ TOOLS**: Simple functions that perform specific tasks (StockPriceTool, CompanyInfoTool, etc.)
- **🤖 AGENTS**: Intelligent entities that can think, plan, and use tools
- **🧠 AGENTIC FUNCTIONS**: What makes agents intelligent (planning, tool usage, reflection, learning)
- **🔄 WORKFLOWS**: How agents coordinate and process information

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

- **LangChain**: Agent framework and tool integration
- **LangGraph**: Workflow orchestration and state management
- **OpenAI GPT**: Large language model for analysis
- **Yahoo Finance API**: Real-time financial data
- **Python**: Core implementation language

## 📁 Project Structure

```
aai-520-group-3-final-project/
├── agents/
│   ├── investment_agent.py      # Main autonomous agent (InvestmentResearchAgent)
│   └── specialist_agents/       # Specialist agents for specific analysis types
│       ├── news_agent.py        # NewsSpecialistAgent
│       ├── earnings_agent.py    # EarningsSpecialistAgent
│       └── market_agent.py      # MarketSpecialistAgent
├── workflows/
│   ├── prompt_chaining.py       # News analysis pipeline
│   ├── routing.py               # Routing workflow (coordinates specialist agents)
│   └── evaluator_optimizer.py   # Quality evaluation & refinement
├── tools/
│   ├── data_sources.py          # Yahoo Finance API integration
│   └── langchain_tools.py       # 5 LangChain tools (Stock, Company, News, Financial, Technical)
├── config.py                    # Configuration management
├── main.py                      # Main entry point (interactive mode)
├── demo_notebook.ipynb          # Jupyter notebook demo
├── .env                         # API keys
└── requirements.txt             # Dependencies
```

## 🎮 Usage Examples

### Agent Functions Demo
```python
from agents.investment_agent import InvestmentResearchAgent

agent = InvestmentResearchAgent()
result = agent.research_stock("AAPL", "comprehensive")
```

### Workflow Patterns Demo
```python
from workflows.prompt_chaining import PromptChainingWorkflow
from workflows.routing import RoutingWorkflow
from workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
from agents.specialist_agents import NewsSpecialistAgent, EarningsSpecialistAgent, MarketSpecialistAgent

# Prompt chaining
pc_workflow = PromptChainingWorkflow()
result = pc_workflow.execute_workflow("AAPL", 5)

# Routing workflow (coordinates specialist agents)
routing_workflow = RoutingWorkflow()
specialists = routing_workflow.route_research_request("AAPL", "comprehensive")
result = routing_workflow.execute_specialist_analysis("AAPL", specialists)

# Direct specialist agent usage
news_agent = NewsSpecialistAgent()
news_result = news_agent.analyze("AAPL")

# Evaluator-optimizer
eo_workflow = EvaluatorOptimizerWorkflow()
result = eo_workflow.execute_workflow("AAPL", "comprehensive", 3)
```

## 🔍 Key Features

### Autonomous Agent Functions
- ✅ **Research Planning**: Autonomous step-by-step research planning
- ✅ **Dynamic Tool Usage**: Intelligent tool selection and execution
- ✅ **Self-Reflection**: Quality assessment of own outputs
- ✅ **Learning System**: Continuous improvement from past analyses

### Multi-Agent Workflows
- ✅ **Prompt Chaining**: Sequential news processing pipeline
- ✅ **Routing**: Intelligent specialist agent coordination
- ✅ **Evaluator-Optimizer**: Quality-driven iterative improvement

### Real-time Analysis
- ✅ **Stock Price Data**: Current prices, changes, volume
- ✅ **Company Information**: Fundamentals, sector, industry
- ✅ **News Analysis**: Sentiment analysis and trend identification
- ✅ **Investment Recommendations**: Data-driven buy/sell/hold suggestions

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
