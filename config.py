"""
Configuration for Multi-Agent Financial Analysis System
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration settings for the financial analysis system."""
    
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    DEFAULT_MODEL = "gpt-3.5-turbo"
    TEMPERATURE = 0.1
    MAX_TOKENS = 4000
    
    # System Configuration
    MAX_ITERATIONS = 3
    MAX_ARTICLES = 10
    DEFAULT_PERIOD = "6mo"
    
    # Learning Configuration
    LEARNING_FILE = "agent_learning.json"
    INSIGHTS_FILE = "agent_insights.json"
    MEMORY_WINDOW = 10
    
    @classmethod
    def validate(cls):
        """Validate required configuration."""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it in your .env file.")
        return True
