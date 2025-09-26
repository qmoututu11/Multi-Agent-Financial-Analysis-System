"""
Prompt Chaining Workflow: Ingest News → Preprocess → Classify → Extract → Summarize
"""

from typing import Dict, Any, List
from datetime import datetime
import re

from tools.data_sources import YahooFinanceAPI
from config import Config

class PromptChainingWorkflow:
    """
    Prompt Chaining Workflow:
    Ingest News → Preprocess → Classify → Extract → Summarize
    """
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
    
    def execute_workflow(self, symbol: str, max_articles: int = 5) -> Dict[str, Any]:
        """Execute the complete prompt chaining workflow."""
        print(f"\n Prompt Chaining Workflow: {symbol}")
        print("=" * 40)
        
        try:
            # Step 1: Ingest News
            print("Step 1: Ingesting news...")
            news_data = self.ingest_news(symbol, max_articles)
            
            if not news_data or news_data.get("status") != "success":
                return {
                    "symbol": symbol,
                    "status": "error",
                    "error": "Failed to ingest news data",
                    "timestamp": datetime.now().isoformat()
                }
            
            # Step 2: Preprocess
            print("Step 2: Preprocessing news data...")
            processed_data = self.preprocess(news_data["articles"])
            
            # Step 3: Classify
            print("Step 3: Classifying sentiment...")
            classified_data = self.classify(processed_data)
            
            # Step 4: Extract
            print("Step 4: Extracting entities...")
            extracted_data = self.extract(classified_data)
            
            # Step 5: Summarize
            print("Step 5: Summarizing results...")
            summarized_data = self.summarize(extracted_data)
            
            return {
                "symbol": symbol,
                "status": "success",
                "workflow": "prompt_chaining",
                "results": summarized_data,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Workflow error: {str(e)}")
            return {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def ingest_news(self, symbol: str, max_articles: int = 5) -> Dict[str, Any]:
        """Step 1: Ingest news data from Yahoo Finance."""
        try:
            result = self.yahoo_api.get_news(symbol, max_articles)
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def preprocess(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Preprocess news data."""
        processed_articles = []
        
        for article in news_data:
            processed_article = {
                "title": self._clean_text(article.get("title", "")),
                "summary": self._clean_text(article.get("summary", "")),
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "key_phrases": self._extract_key_phrases(
                    article.get("title", "") + " " + article.get("summary", "")
                )
            }
            processed_articles.append(processed_article)
        
        return processed_articles
    
    def classify(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Classify sentiment of processed data."""
        classified_articles = []
        
        for article in processed_data:
            # Simple sentiment classification
            sentiment = self._classify_sentiment(article["title"] + " " + article["summary"])
            
            classified_article = {
                **article,
                "sentiment": sentiment
            }
            classified_articles.append(classified_article)
        
        return classified_articles
    
    def extract(self, classified_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 4: Extract entities from classified data."""
        extracted_articles = []
        
        for article in classified_data:
            # Extract basic entities
            entities = self._extract_entities(article["title"] + " " + article["summary"])
            
            extracted_article = {
                **article,
                "entities": entities
            }
            extracted_articles.append(extracted_article)
        
        return extracted_articles
    
    def summarize(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 5: Summarize the complete analysis."""
        if not extracted_data:
            return {"summary": "No data to summarize", "articles_processed": 0}
        
        # Calculate overall sentiment
        total_articles = len(extracted_data)
        positive_count = sum(1 for article in extracted_data 
                           if article["sentiment"]["overall"] == "positive")
        negative_count = sum(1 for article in extracted_data 
                           if article["sentiment"]["overall"] == "negative")
        neutral_count = total_articles - positive_count - negative_count
        
        # Extract all entities
        all_entities = {"companies": [], "numbers": [], "percentages": []}
        for article in extracted_data:
            for entity_type, entities in article["entities"].items():
                all_entities[entity_type].extend(entities)
        
        # Remove duplicates
        for entity_type in all_entities:
            all_entities[entity_type] = list(set(all_entities[entity_type]))
        
        return {
            "articles_processed": total_articles,
            "sentiment_distribution": {
                "positive": positive_count,
                "negative": negative_count,
                "neutral": neutral_count,
                "overall_sentiment": "positive" if positive_count > negative_count 
                                   else "negative" if negative_count > positive_count 
                                   else "neutral"
            },
            "key_entities": all_entities,
            "top_articles": [
                {
                    "title": article["title"][:100] + "..." if len(article["title"]) > 100 else article["title"],
                    "sentiment": article["sentiment"]["overall"]
                }
                for article in extracted_data[:3]
            ]
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?;:-]', '', text)
        
        return text.strip()
    
    def _extract_key_phrases(self, text: str, max_phrases: int = 5) -> List[str]:
        """Extract key phrases from text."""
        if not text:
            return []
        
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequency
        word_freq = {}
        for word in filtered_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:max_phrases]]
    
    def _classify_sentiment(self, text: str) -> Dict[str, Any]:
        """Classify sentiment of text."""
        if not text:
            return {"overall": "neutral", "score": 0.0}
        
        text_lower = text.lower()
        
        positive_words = ['good', 'great', 'excellent', 'positive', 'growth', 'profit', 'gain', 'rise', 'increase', 'strong']
        negative_words = ['bad', 'terrible', 'negative', 'decline', 'loss', 'fall', 'drop', 'decrease', 'weak']
        
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return {"overall": "positive", "score": positive_count / (positive_count + negative_count)}
        elif negative_count > positive_count:
            return {"overall": "negative", "score": negative_count / (positive_count + negative_count)}
        else:
            return {"overall": "neutral", "score": 0.0}
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract basic entities from text."""
        entities = {
            "companies": [],
            "numbers": [],
            "percentages": []
        }
        
        # Extract numbers
        numbers = re.findall(r'\b\d+(?:,\d{3})*(?:\.\d+)?\b', text)
        entities["numbers"] = numbers
        
        # Extract percentages
        percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', text)
        entities["percentages"] = percentages
        
        # Extract potential company names (simple pattern)
        companies = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|Company|Ltd)\b', text)
        entities["companies"] = companies
        
        return entities
