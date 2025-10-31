#!/usr/bin/env python3
"""
FastAPI REST API for Multi-Agent Financial Analysis System
Exposes LangGraph orchestration as REST endpoints
"""

import sys
import os
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflows.langgraph_orchestration import LangGraphOrchestrator
from config import Config

# Initialize FastAPI app
app = FastAPI(
    title="Multi-Agent Financial Analysis API",
    description="REST API for LangGraph-based financial analysis orchestration",
    version="1.0.0"
)

# CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your React app URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize orchestrator
orchestrator = None

@app.on_event("startup")
async def startup_event():
    """Initialize the orchestrator on startup."""
    global orchestrator
    try:
        Config.validate()
        orchestrator = LangGraphOrchestrator()
        print("✅ API initialized successfully")
    except Exception as e:
        print(f"❌ API initialization error: {e}")
        raise


# Request/Response models
class AnalysisRequest(BaseModel):
    """Request model for analysis."""
    symbol: str = Field(..., description="Stock symbol to analyze", example="AAPL")
    workflow_type: str = Field(
        default="comprehensive",
        description="Type of workflow: comprehensive, routing, investment_agent, prompt_chaining, evaluator_optimizer",
        example="comprehensive"
    )
    focus: str = Field(
        default="comprehensive",
        description="Analysis focus: comprehensive, news, earnings, technical, market",
        example="comprehensive"
    )


class AnalysisResponse(BaseModel):
    """Response model for analysis."""
    status: str
    symbol: str
    workflow_type: str
    result: Dict[str, Any]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str


# API Endpoints

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "message": "Multi-Agent Financial Analysis API is running"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return {
        "status": "healthy",
        "message": "API is operational"
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest):
    """
    Execute LangGraph orchestration workflow.
    
    Args:
        request: AnalysisRequest with symbol, workflow_type, and focus
    
    Returns:
        AnalysisResponse with complete analysis results
    """
    if orchestrator is None:
        raise HTTPException(
            status_code=503,
            detail="Orchestrator not initialized. Please check API configuration."
        )
    
    try:
        # Validate workflow_type
        valid_workflows = [
            "comprehensive", "routing", "investment_agent",
            "prompt_chaining", "evaluator_optimizer"
        ]
        if request.workflow_type not in valid_workflows:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid workflow_type. Must be one of: {', '.join(valid_workflows)}"
            )
        
        # Validate focus
        valid_focuses = ["comprehensive", "news", "earnings", "technical", "market"]
        if request.focus not in valid_focuses:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid focus. Must be one of: {', '.join(valid_focuses)}"
            )
        
        # Execute workflow
        result = orchestrator.run(
            symbol=request.symbol.upper(),
            focus=request.focus,
            workflow_type=request.workflow_type
        )
        
        # Add timestamp
        from datetime import datetime
        result["timestamp"] = datetime.now().isoformat()
        
        return AnalysisResponse(
            status=result.get("status", "success"),
            symbol=request.symbol.upper(),
            workflow_type=request.workflow_type,
            result=result.get("result", {}),
            timestamp=result["timestamp"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis error: {str(e)}"
        )


@app.get("/api/workflows")
async def get_workflows():
    """Get list of available workflows."""
    return {
        "workflows": [
            {
                "id": "comprehensive",
                "name": "Comprehensive Analysis",
                "description": "Routes to all specialist agents (news, earnings, market)"
            },
            {
                "id": "routing",
                "name": "Routing Workflow",
                "description": "Routes to specialist agents based on focus"
            },
            {
                "id": "investment_agent",
                "name": "Investment Agent",
                "description": "Uses main autonomous investment research agent"
            },
            {
                "id": "prompt_chaining",
                "name": "Prompt Chaining",
                "description": "Executes news analysis pipeline (ingest → preprocess → classify → extract → summarize)"
            },
            {
                "id": "evaluator_optimizer",
                "name": "Evaluator-Optimizer",
                "description": "Generate → Evaluate → Optimize workflow with quality feedback"
            }
        ]
    }


@app.get("/api/focus-types")
async def get_focus_types():
    """Get list of available focus types."""
    return {
        "focus_types": [
            {"id": "comprehensive", "name": "Comprehensive", "description": "Full analysis"},
            {"id": "news", "name": "News", "description": "News sentiment analysis"},
            {"id": "earnings", "name": "Earnings", "description": "Financial and earnings analysis"},
            {"id": "technical", "name": "Technical", "description": "Technical indicators and market trends"},
            {"id": "market", "name": "Market", "description": "Market data and price trends"}
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

