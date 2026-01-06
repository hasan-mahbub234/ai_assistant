import logging
import uuid
from typing import Optional, Dict, List
from src.core.embeddings import EmbeddingManager
from src.core.pinecone_manager import PineconeManager
from src.core.database_manager import DatabaseManager
from src.ai.language_detector import LanguageDetector
from src.ai.llm_manager import LLMManager
from src.ai.context_builder import ContextBuilder
from src.ai.prompt_builder import PromptBuilder
from src.config import config

logger = logging.getLogger(__name__)


class AIService:
    """Main AI customer support service"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize AI service components"""
        logger.info("Initializing AI Service...")
        
        # Core components
        self.embeddings = EmbeddingManager()
        self.pinecone = PineconeManager()
        self.db = DatabaseManager()
        
        # AI components
        self.language_detector = LanguageDetector()
        self.llm = LLMManager(api_key=config.GROQ_API_KEY)
        self.context_builder = ContextBuilder()
        self.prompt_builder = PromptBuilder()
        
        # State
        self.conversation_history = []
        
        logger.info("✅ AI Service initialized successfully")
    
    def search_vector_db(self, query: str, top_k: int = 5) -> List[Dict]:
        """Search Pinecone vector database"""
        try:
            # Get query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search Pinecone
            results = self.pinecone.search(
                query_embedding=query_embedding,
                top_k=top_k
            )
            
            # Format results
            documents = []
            for match in results.matches:
                metadata = match.metadata
                documents.append({
                    'id': match.id,
                    'score': match.score,
                    'name': metadata.get('name', 'Unknown Product'),
                    'brand': metadata.get('brand', ''),
                    'price': metadata.get('price', 0.0),
                    'category': metadata.get('category', ''),
                    'average_rating': metadata.get('average_rating', 0.0),
                    'review_count': metadata.get('review_count', 0),
                    'description': metadata.get('_preview', ''),
                    'tags': metadata.get('tags', []),
                    'free_delivery': metadata.get('free_delivery', False),
                    'best_selling': metadata.get('best_selling', False)
                })
            
            logger.info(f"Found {len(documents)} products in vector database")
            return documents
            
        except Exception as e:
            logger.error(f"Vector database search error: {e}")
            return []
    
    def get_response(self, user_query: str, user_id: Optional[int] = None,
                    conversation_id: Optional[str] = None) -> Dict:
        """Generate AI response for user query"""
        try:
            # Generate or use conversation ID
            if not conversation_id:
                conversation_id = self._generate_conversation_id(user_id)
            
            # Detect language
            language = self.language_detector.detect(user_query)
            logger.info(f"Language detected: {language}")
            
            # Search for relevant products
            vector_results = self.search_vector_db(user_query, top_k=3)
            
            # Fallback to database search if no vector results
            if not vector_results:
                logger.info("No vector results, falling back to database search")
                db_results = self.db.search_products(user_query, limit=3)
                
                # Convert DB format to match vector results format
                vector_results = [
                    {
                        'id': f"db_{p['product_id']}",
                        'score': 0.8,  # Default score for DB results
                        'name': p['name'],
                        'brand': p['brand'],
                        'price': p['price'],
                        'category': p['category'],
                        'average_rating': 0.0,
                        'review_count': 0,
                        'description': p['description'],
                        'tags': p['tags'].split(',') if p['tags'] else [],
                        'free_delivery': False,
                        'best_selling': False
                    }
                    for p in db_results
                ]
            
            # Build context
            context = self.context_builder.build(
                products=vector_results,
                query=user_query,
                language=language
            )
            
            # Build prompt
            messages = self.prompt_builder.build(
                query=user_query,
                context=context,
                language=language
            )
            
            # Get LLM response
            response_text = self.llm.generate(messages)
            
            # Format response
            formatted_response = self.llm.format_response(response_text)
            
            # Store conversation
            self._store_conversation(
                conversation_id=conversation_id,
                user_id=user_id,
                query=user_query,
                response=formatted_response,
                language=language
            )
            
            return {
                'answer': formatted_response,
                'language': language,
                'conversation_id': conversation_id,
                'products': vector_results,
                'sources': 'vector_db' if vector_results else 'database'
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'answer': "I apologize, but I encountered an error. Please try again.",
                'language': 'english',
                'conversation_id': conversation_id or 'error',
                'products': [],
                'error': str(e)
            }
    
    def _generate_conversation_id(self, user_id: Optional[int] = None) -> str:
        """Generate unique conversation ID"""
        if user_id:
            return f"user_{user_id}_{uuid.uuid4().hex[:8]}"
        return f"guest_{uuid.uuid4().hex[:12]}"
    
    def _store_conversation(self, conversation_id: str, user_id: Optional[int],
                           query: str, response: str, language: str):
        """Store conversation in history"""
        conversation = {
            'conversation_id': conversation_id,
            'user_id': user_id,
            'query': query,
            'response': response,
            'language': language,
            'timestamp': str(uuid.uuid1())
        }
        
        self.conversation_history.append(conversation)
        
        # Keep only last 100 conversations
        if len(self.conversation_history) > 100:
            self.conversation_history = self.conversation_history[-100:]
    
    def get_pinecone_status(self):
        """Get Pinecone connection status"""
        return self.pinecone.check_connection()
    
    def get_product_recommendations(self, keyword: str, limit: int = 5):
        """Get product recommendations with reviews"""
        return self.db.get_product_with_reviews(keyword, limit)
    
    def track_order(self, order_id: str):
        """Track order information"""
        return self.db.get_order_info(order_id)