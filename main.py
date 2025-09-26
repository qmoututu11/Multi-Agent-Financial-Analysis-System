#!/usr/bin/env python3
"""
Multi-Agent Financial Analysis System - Interactive Demo
AAI-520 Group 3 Final Project
"""

import sys
import os
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.investment_agent import InvestmentResearchAgent
from workflows.prompt_chaining import PromptChainingWorkflow
from workflows.routing import RoutingWorkflow
from workflows.evaluator_optimizer import EvaluatorOptimizerWorkflow
from config import Config

class MultiAgentFinancialAnalysisSystem:
    """Multi-Agent Financial Analysis System for Interactive Demo."""
    
    def __init__(self):
        """Initialize the multi-agent system."""
        try:
            Config.validate()
            print("Configuration validated successfully")
        except ValueError as e:
            print(f"Configuration error: {e}")
            return
        
        # Initialize components
        self.agent = InvestmentResearchAgent()
        self.prompt_chaining_workflow = PromptChainingWorkflow()
        self.routing_workflow = RoutingWorkflow()
        self.evaluator_optimizer_workflow = EvaluatorOptimizerWorkflow()
        
        print("Multi-Agent Financial Analysis System initialized")
    
    def analyze_agent_functions(self, symbol: str) -> Dict[str, Any]:
        """Analyze using autonomous agent functions."""
        print(f"\nAnalyzing {symbol} with Autonomous Agent Functions")
        print("=" * 50)
        
        try:
            result = self.agent.research_stock(symbol, "comprehensive")
            
            if result["status"] == "success":
                print("Agent Analysis Completed Successfully!")
                print(f"Plan: {result['plan'][:100]}...")
                print(f"Analysis: {result['analysis'][:200]}...")
                print(f"Reflection: {result['reflection']['reflection'][:100]}...")
                return result
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def analyze_prompt_chaining(self, symbol: str) -> Dict[str, Any]:
        """Analyze using prompt chaining workflow."""
        print(f"\nAnalyzing {symbol} with Prompt Chaining Workflow")
        print("=" * 50)
        
        try:
            result = self.prompt_chaining_workflow.execute_workflow(symbol, 5)
            
            if result["status"] == "success":
                print("Prompt Chaining Analysis Completed Successfully!")
                results = result["results"]
                print(f"Articles Processed: {results['articles_processed']}")
                print(f"Overall Sentiment: {results['sentiment_distribution']['overall_sentiment']}")
                print(f"Key Entities: {list(results['key_entities'].keys())}")
                return result
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def analyze_routing(self, symbol: str) -> Dict[str, Any]:
        """Analyze using routing workflow."""
        print(f"\nAnalyzing {symbol} with Routing Workflow")
        print("=" * 50)
        
        try:
            specialists = self.routing_workflow.route_research_request(symbol, "comprehensive")
            result = self.routing_workflow.execute_specialist_analysis(symbol, specialists)
            
            if result["status"] == "success":
                print("Routing Workflow Analysis Completed Successfully!")
                print(f"Specialists Used: {', '.join(result['specialists_used'])}")
                print(f"Combined Summary:\n{result['combined_summary'][:300]}...")
                return result
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def analyze_evaluator_optimizer(self, symbol: str) -> Dict[str, Any]:
        """Analyze using evaluator-optimizer workflow."""
        print(f"\nAnalyzing {symbol} with Evaluator-Optimizer Workflow")
        print("=" * 50)
        
        try:
            result = self.evaluator_optimizer_workflow.execute_workflow(symbol, "comprehensive", 2)
            
            if result["status"] == "success":
                print("Evaluator-Optimizer Analysis Completed Successfully!")
                print(f"Initial Quality: {result['initial_evaluation']['overall_score']:.2f}")
                print(f"Final Quality: {result['final_evaluation']['overall_score']:.2f}")
                print(f"Iterations: {result['iterations_completed']}")
                print(f"Final Analysis: {result['final_analysis'][:200]}...")
                return result
            else:
                print(f"Error: {result.get('error', 'Unknown error')}")
                return result
                
        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {"status": "error", "error": str(e)}
    
    def run_comprehensive_analysis(self, symbol: str):
        """Run comprehensive analysis using all workflows."""
        print(f"\nCOMPREHENSIVE MULTI-AGENT ANALYSIS: {symbol}")
        print("=" * 60)
        
        workflows = [
            ("Agent Functions", self.analyze_agent_functions),
            ("Prompt Chaining", self.analyze_prompt_chaining),
            ("Routing Workflow", self.analyze_routing),
            ("Evaluator-Optimizer", self.analyze_evaluator_optimizer)
        ]
        
        results = {}
        
        for workflow_name, workflow_func in workflows:
            try:
                result = workflow_func(symbol)
                results[workflow_name] = result["status"]
                print(f"{workflow_name}: {'Success' if result['status'] == 'success' else 'Failed'}")
            except Exception as e:
                print(f"{workflow_name}: Error - {str(e)}")
                results[workflow_name] = "error"
        
        # Summary
        print(f"\nANALYSIS SUMMARY")
        print("=" * 30)
        for workflow_name, status in results.items():
            print(f"{workflow_name}: {status}")
        
        success_count = sum(1 for status in results.values() if status == "success")
        total_count = len(results)
        print(f"\nSuccess Rate: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
        
        return results
    
    def show_learning_summary(self):
        """Show learning summary."""
        print("\nLearning Summary:")
        print(self.agent.get_learning_summary())
    
    def interactive_analysis(self):
        """Interactive analysis mode."""
        print("\nInteractive Multi-Agent Financial Analysis")
        print("=" * 50)
        print("Available commands:")
        print("  'agent <symbol>' - Analyze with agent functions")
        print("  'prompt <symbol>' - Analyze with prompt chaining")
        print("  'routing <symbol>' - Analyze with routing workflow")
        print("  'evaluator <symbol>' - Analyze with evaluator-optimizer")
        print("  'all <symbol>' - Run comprehensive analysis")
        print("  'learning' - Show learning summary")
        print("  'quit' - Exit")
        
        while True:
            try:
                user_input = input("\nEnter command: ").strip().lower()
                
                if user_input == 'quit':
                    print("Goodbye!")
                    break
                elif user_input == 'learning':
                    self.show_learning_summary()
                elif user_input.startswith('agent '):
                    symbol = user_input.split(' ', 1)[1].upper()
                    self.analyze_agent_functions(symbol)
                elif user_input.startswith('prompt '):
                    symbol = user_input.split(' ', 1)[1].upper()
                    self.analyze_prompt_chaining(symbol)
                elif user_input.startswith('routing '):
                    symbol = user_input.split(' ', 1)[1].upper()
                    self.analyze_routing(symbol)
                elif user_input.startswith('evaluator '):
                    symbol = user_input.split(' ', 1)[1].upper()
                    self.analyze_evaluator_optimizer(symbol)
                elif user_input.startswith('all '):
                    symbol = user_input.split(' ', 1)[1].upper()
                    self.run_comprehensive_analysis(symbol)
                else:
                    print("Unknown command. Type 'quit' to exit.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {str(e)}")

def main():
    """Main function - entry point for the application."""
    print("Multi-Agent Financial Analysis System")
    print("AAI-520 Group 3 Final Project")
    print("=" * 50)
    
    try:
        system = MultiAgentFinancialAnalysisSystem()
        
        # For notebook compatibility, just start interactive mode
        system.interactive_analysis()
            
    except Exception as e:
        print(f"System error: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()