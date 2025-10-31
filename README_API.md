# API Documentation

## FastAPI REST API Setup

The system exposes LangGraph orchestration through a FastAPI REST API.

### Quick Start

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Start the API server:**
```bash
python3 api.py
```

Or with uvicorn directly:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

3. **API will be available at:**
- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs` (Swagger UI)
- ReDoc: `http://localhost:8000/redoc`

## API Endpoints

### POST `/api/analyze`

Execute LangGraph orchestration workflow.

**Request Body:**
```json
{
  "symbol": "AAPL",
  "workflow_type": "comprehensive",
  "focus": "comprehensive"
}
```

**Response:**
```json
{
  "status": "success",
  "symbol": "AAPL",
  "workflow_type": "comprehensive",
  "result": {
    "symbol": "AAPL",
    "workflow_type": "comprehensive",
    "nodes_executed": ["start", "route", "news_specialist", "earnings_specialist", "market_specialist", "combine_results", "finalize"],
    "results": { ... },
    "status": "success"
  },
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET `/api/workflows`

Get list of available workflows.

**Response:**
```json
{
  "workflows": [
    {
      "id": "comprehensive",
      "name": "Comprehensive Analysis",
      "description": "Routes to all specialist agents"
    },
    ...
  ]
}
```

### GET `/api/focus-types`

Get list of available focus types.

**Response:**
```json
{
  "focus_types": [
    {
      "id": "comprehensive",
      "name": "Comprehensive",
      "description": "Full analysis"
    },
    ...
  ]
}
```

### GET `/health`

Health check endpoint.

## React Frontend Integration

### Example React Component

See `frontend/AnalysisComponent.jsx` for a complete example.

### Basic Usage

```javascript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Execute analysis
const response = await axios.post(`${API_BASE_URL}/api/analyze`, {
  symbol: 'AAPL',
  workflow_type: 'comprehensive',
  focus: 'comprehensive'
});

console.log(response.data);
```

### CORS Configuration

The API includes CORS middleware. In production, update the `allow_origins` in `api.py` to specify your React app's URL:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your React app URL
    ...
)
```

## Workflow Types

- `comprehensive`: Routes to all specialist agents
- `routing`: Routes to specialists based on focus
- `investment_agent`: Uses main investment agent
- `prompt_chaining`: News analysis pipeline
- `evaluator_optimizer`: Generate → Evaluate → Optimize

## Focus Types

- `comprehensive`: Full analysis
- `news`: News sentiment analysis
- `earnings`: Financial and earnings analysis
- `technical`: Technical indicators
- `market`: Market data and trends

## Error Handling

All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid parameters)
- `500`: Server error
- `503`: Service unavailable (orchestrator not initialized)

