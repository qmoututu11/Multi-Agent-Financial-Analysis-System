"""
Prompt Chaining Workflow: Ingest News → Preprocess → Classify → Extract → Summarize
Uses LLM prompts at each step for intelligent processing
"""

from typing import Dict, Any, List
from datetime import datetime
import re

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

from tools.data_sources import YahooFinanceAPI
from config import Config

class PromptChainingWorkflow:
    """
    Prompt Chaining Workflow with LLM-powered steps:
    Ingest News → Preprocess → Classify → Extract → Summarize
    Each step uses LLM prompts for intelligent processing
    """
    
    def __init__(self):
        self.yahoo_api = YahooFinanceAPI()
        
        # Initialize LLM for prompt chaining
        self.llm = ChatOpenAI(
            model=Config.DEFAULT_MODEL,
            temperature=Config.TEMPERATURE,
            api_key=Config.OPENAI_API_KEY
        )
    
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
            
            # Log articles being ingested
            if result.get("status") == "success":
                articles = result.get("articles", [])
                print(f"  Fetched {len(articles)} articles")
                for i, article in enumerate(articles, 1):
                    title = article.get("title", "No title")
                    summary = article.get("summary", "")
                    url = article.get("url", "")
                    source = article.get("source", "")
                    
                    print(f"\n  Article {i}:")
                    print(f"    Title: {title}")
                    if summary:
                        print(f"    Summary: {summary[:200]}{'...' if len(summary) > 200 else ''}")
                    else:
                        print(f"    Summary: [No summary available]")
                    if url:
                        print(f"    Source: {source} ({url[:50]}...)")
                    elif source:
                        print(f"    Source: {source}")
            
            return result
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def preprocess(self, news_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Preprocess news data using LLM to extract key information."""
        preprocessing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a news preprocessing assistant. Extract and structure key information from financial news articles.

For each article, identify:
- Clean title (remove special characters, normalize)
- Key summary points (3-5 bullet points)
- Key financial terms and numbers mentioned
- Market impact indicators

Return structured information that highlights the most important financial aspects."""),
            ("human", """Preprocess this news article:

Title: {title}
Summary: {summary}

Extract and structure the key financial information. Return a brief, clean summary focusing on financial metrics, market impact, and key facts.""")
        ])
        
        processed_articles = []
        
        for article in news_data:
            title = self._clean_text(article.get("title", ""))
            original_summary = self._clean_text(article.get("summary", ""))
            
            # For LLM preprocessing, use original summary if available, otherwise use title
            summary_for_llm = original_summary if original_summary else (title if title else "No content available")
            
            # Use LLM for intelligent preprocessing
            try:
                chain = preprocessing_prompt | self.llm
                response = chain.invoke({
                    "title": title,
                    "summary": summary_for_llm
                })
                llm_preprocessed = response.content.strip()
            except Exception as e:
                print(f"  LLM preprocessing error: {e}, using fallback")
                llm_preprocessed = summary_for_llm
            
            processed_article = {
                "title": title,
                "summary": llm_preprocessed,  # LLM-processed version
                "original_summary": original_summary,  # Original from Yahoo Finance (may be empty)
                "url": article.get("url", ""),
                "source": article.get("source", ""),
                "key_phrases": self._extract_key_phrases(title + " " + (original_summary or title))
            }
            processed_articles.append(processed_article)
        
        return processed_articles
    
    def classify(self, processed_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Classify sentiment using LLM for intelligent analysis."""
        sentiment_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial sentiment analyst. Analyze news articles for their sentiment toward a stock.

Classify the sentiment as:
- "positive": Bullish, growth, gains, positive outlook
- "negative": Bearish, decline, losses, negative outlook  
- "neutral": Mixed, balanced, no clear direction

Also provide:
- Confidence score (0.0 to 1.0)
- Key reasons for the sentiment
- Market impact assessment (low, medium, high)

Respond in JSON format: {{"overall": "positive|negative|neutral", "confidence": 0.8, "reasons": ["reason1", "reason2"], "market_impact": "medium"}}"""),
            ("human", """Analyze sentiment for this financial news article:

Title: {title}
Summary: {summary}

Classify the sentiment and provide analysis.""")
        ])
        
        classified_articles = []
        
        for article in processed_data:
            # Use LLM for intelligent sentiment classification
            try:
                chain = sentiment_prompt | self.llm
                response = chain.invoke({
                    "title": article.get("title", ""),
                    "summary": article.get("summary", "")
                })
                
                # Parse LLM response (try JSON, fallback to text parsing)
                import json
                try:
                    sentiment_result = json.loads(response.content.strip())
                except:
                    # Fallback to text parsing
                    sentiment_result = self._parse_sentiment_response(response.content)
                    
            except Exception as e:
                print(f"LLM sentiment classification error: {e}, using fallback")
                sentiment_result = self._classify_sentiment(article.get("title", "") + " " + article.get("summary", ""))
            
            classified_article = {
                **article,
                "sentiment": sentiment_result
            }
            classified_articles.append(classified_article)
        
        return classified_articles
    
    def _parse_sentiment_response(self, response_text: str) -> Dict[str, Any]:
        """Parse sentiment response from LLM if not valid JSON."""
        response_lower = response_text.lower()
        
        # Extract sentiment
        if "positive" in response_lower or "bullish" in response_lower:
            overall = "positive"
        elif "negative" in response_lower or "bearish" in response_lower:
            overall = "negative"
        else:
            overall = "neutral"
        
        # Extract confidence (look for numbers)
        import re
        confidence_match = re.search(r'0\.\d+|1\.0', response_text)
        confidence = float(confidence_match.group()) if confidence_match else 0.5
        
        return {
            "overall": overall,
            "confidence": confidence,
            "reasons": ["LLM analysis"],
            "market_impact": "medium"
        }
    
    def extract(self, classified_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 4: Extract entities using LLM for intelligent entity extraction."""
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial entity extraction specialist. Extract key entities from financial news.

Extract:
- Companies mentioned (ticker symbols and company names)
- Financial metrics (revenue, earnings, growth rates, percentages)
- Key people (executives, analysts)
- Market indicators (prices, volumes, market cap)
- Dates and timeframes

Return in JSON format:
{{"companies": ["AAPL", "Apple Inc."], "numbers": ["$150B revenue", "15% growth"], "percentages": ["15%", "20%"], "people": ["CEO Tim Cook"], "dates": ["Q1 2024"]}}"""),
            ("human", """Extract entities from this financial news:

Title: {title}
Summary: {summary}

Extract all relevant financial entities.""")
        ])
        
        extracted_articles = []
        
        for article in classified_data:
            # Use LLM for intelligent entity extraction
            try:
                chain = extraction_prompt | self.llm
                response = chain.invoke({
                    "title": article.get("title", ""),
                    "summary": article.get("summary", "")
                })
                
                # Parse LLM response
                import json
                try:
                    entities = json.loads(response.content.strip())
                except:
                    # Fallback to regex extraction
                    entities = self._extract_entities(article.get("title", "") + " " + article.get("summary", ""))
                    
            except Exception as e:
                print(f"LLM entity extraction error: {e}, using fallback")
                entities = self._extract_entities(article.get("title", "") + " " + article.get("summary", ""))
            
            extracted_article = {
                **article,
                "entities": entities
            }
            extracted_articles.append(extracted_article)
        
        return extracted_articles
    
    def summarize(self, extracted_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Step 5: Summarize using LLM for intelligent synthesis."""
        if not extracted_data:
            return {"summary": "No data to summarize", "articles_processed": 0}
        
        # Prepare context for LLM summarization - use original summaries for full context
        articles_context = []
        for article in extracted_data:
            # Use original_summary if available (from preprocessing step), otherwise use summary
            article_summary = article.get("original_summary", article.get("summary", ""))
            if not article_summary:
                # Fallback: try to get summary from the article
                article_summary = article.get("summary", "")
            
            # If still no summary, use title as minimum content
            if not article_summary:
                article_summary = article.get("title", "No content available")
            
            articles_context.append({
                "title": article.get("title", ""),
                "summary": article_summary,  # Use original summary for full context
                "sentiment": article.get("sentiment", {}),
                "entities": article.get("entities", {})
            })
        
        # Calculate basic statistics
        total_articles = len(extracted_data)
        positive_count = sum(1 for article in extracted_data 
                           if article.get("sentiment", {}).get("overall") == "positive")
        negative_count = sum(1 for article in extracted_data 
                           if article.get("sentiment", {}).get("overall") == "negative")
        neutral_count = total_articles - positive_count - negative_count
        
        # Use LLM for intelligent summary
        summary_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a financial news analyst. Synthesize multiple news articles into a comprehensive summary.

Analyze the articles and provide:
1. Overall market sentiment and trend
2. Key financial metrics and numbers mentioned
3. Major companies and entities involved
4. Market impact assessment
5. Key takeaways for investors

Provide a clear, concise summary that highlights the most important information."""),
            ("human", """Synthesize these {count} financial news articles into a comprehensive summary:

{articles_context}

Provide an intelligent summary of the key themes, sentiment, and financial information.""")
        ])
        
        try:
            # Format articles for LLM - include full summaries
            articles_text = "\n\n".join([
                f"Article {i+1}:\nTitle: {art['title']}\nSummary: {art['summary']}\nSentiment: {art.get('sentiment', {}).get('overall', 'unknown')}"
                for i, art in enumerate(articles_context[:10])  # Limit to 10 for context
            ])
            
            # Verify we have content
            if not articles_text.strip():
                print("  Warning: No article content available for summarization")
                llm_summary = f"Processed {total_articles} articles but no content available for summary."
            else:
                chain = summary_prompt | self.llm
                response = chain.invoke({
                    "count": len(articles_context),
                    "articles_context": articles_text
                })
                llm_summary = response.content.strip()
        except Exception as e:
            print(f"  LLM summarization error: {e}, using fallback")
            llm_summary = f"Processed {total_articles} articles. Sentiment: {positive_count} positive, {negative_count} negative, {neutral_count} neutral."
        
        # Extract all entities
        all_entities = {"companies": [], "numbers": [], "percentages": []}
        for article in extracted_data:
            entities = article.get("entities", {})
            if isinstance(entities, dict):
                for entity_type in ["companies", "numbers", "percentages"]:
                    if entity_type in entities:
                        if isinstance(entities[entity_type], list):
                            all_entities[entity_type].extend(entities[entity_type])
        
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
            "llm_summary": llm_summary,  # LLM-generated intelligent summary
            "top_articles": [
                {
                    "title": article.get("title", "")[:100] + "..." if len(article.get("title", "")) > 100 else article.get("title", ""),
                    "sentiment": article.get("sentiment", {}).get("overall", "unknown")
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
