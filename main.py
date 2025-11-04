#!/usr/bin/env python3
"""
Multi-Agent Financial Analysis System
AAI-520 Group 3 Final Project

Single entry point through LangGraph orchestration.
"""

import sys
import os
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from workflows.langgraph_orchestration import LangGraphOrchestrator
from config import Config


class MultiAgentFinancialAnalysisSystem:
    """Multi-Agent Financial Analysis System - Single Entry Point."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        try:
            Config.validate()
            print("Configuration validated successfully")
        except ValueError as e:
            print(f"Configuration error: {e}")
            raise
        
        # Initialize LangGraph orchestrator (single entry point)
        self.orchestrator = LangGraphOrchestrator()
        
        print("Multi-Agent Financial Analysis System initialized")
        print("LangGraph orchestration enabled")
    
    def analyze(self, symbol: str, focus: str = "comprehensive") -> Dict[str, Any]:
        """
        Single entry point for financial analysis.
        
        Args:
            symbol: Stock symbol to analyze
            focus: Analysis focus (comprehensive, news, earnings, technical, market, forecast)
        
        Returns:
            Analysis results dictionary
        """
        print(f"\nAnalyzing {symbol}")
        if focus != "comprehensive":
            print(f"Focus: {focus}")
        print("=" * 50)
        
        try:
            result = self.orchestrator.run(
                symbol=symbol,
                focus=focus,
                workflow_type="comprehensive"
            )
            
            if result.get("status") == "success":
                # Summary already shown in comprehensive analysis
                return result
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f" Error: {error_msg}")
                return result
                
        except Exception as e:
            print(f" Analysis error: {str(e)}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "error": str(e)}
    
    def interactive_analysis(self):
        """Interactive analysis mode."""
        print("\n" + "=" * 60)
        print("Multi-Agent Financial Analysis System")
        print("AAI-520 Group 3 Final Project")
        print("=" * 60)
        print("\nAvailable commands:")
        print("  '<symbol>' - Analyze symbol (all specialists)")
        print("  '<symbol> <focus>' - Analyze with specific focus")
        print("    Focus: news, earnings, technical, market, forecast, comprehensive")
        print("  'help' - Show this help message")
        print("  'quit' or 'exit' - Exit")
        print("\nExamples:")
        print("  AAPL                    # All specialists")
        print("  AAPL news               # All specialists, emphasis on news")
        print("  TSLA earnings           # All specialists, emphasis on earnings")
        print("=" * 60)
        
        while True:
            try:
                user_input = input("\nEnter command: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\nCommands:")
                    print("  <symbol> - Analyze with all specialists")
                    print("  <symbol> <focus> - Analyze with focus (news, earnings, technical, market, forecast)")
                    continue
                
                # Parse command
                parts = user_input.split()
                
                if len(parts) == 1:
                    # Just symbol - use all specialists
                    symbol = parts[0].upper()
                    self.analyze(symbol)
                
                elif len(parts) == 2:
                    # Symbol + focus
                    symbol = parts[0].upper()
                    focus = parts[1].lower()
                    
                    # Validate focus
                    valid_focuses = ['news', 'earnings', 'technical', 'market', 'forecast', 'comprehensive']
                    if focus not in valid_focuses:
                        print(f"Invalid focus: {focus}")
                        print(f"Valid options: {', '.join(valid_focuses)}")
                        continue
                    
                    self.analyze(symbol, focus=focus)
                
                else:
                    print("Invalid command. Use: <symbol> [focus]")
                    print("Examples: AAPL  or  AAPL news")
                    print("Type 'help' for more information")
                    
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f" Error: {str(e)}")
                print("Type 'help' for usage examples")


def main():
    """Main function - single entry point."""
    try:
        system = MultiAgentFinancialAnalysisSystem()
        system.interactive_analysis()
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    except Exception as e:
        print(f" System error: {str(e)}")
        print("Please check your configuration and try again.")
        sys.exit(1)


if __name__ == "__main__":
    main()