from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
from src.ai.ai_service import AIService
from src.ai.chat_handler import ChatHandler

logger = logging.getLogger(__name__)

ai_router = APIRouter()
ai_service = AIService()
chat_handler = ChatHandler()


# Request/Response models
class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    conversation_id: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    language: str
    conversation_id: str
    success: bool = True
    products: List[dict] = []

class ProductRequest(BaseModel):
    search_query: str
    limit: Optional[int] = 5

class OrderRequest(BaseModel):
    order_id: str


@ai_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with AI"""
    try:
        logger.info(f"Chat request: {request.message[:100]}")
        
        # Get AI response
        result = ai_service.get_response(
            user_query=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Store in chat handler
        if request.conversation_id:
            chat_handler.add_message(
                request.conversation_id,
                "user",
                request.message
            )
            chat_handler.add_message(
                request.conversation_id,
                "assistant",
                result['answer']
            )
            chat_handler.update_product_context(
                request.conversation_id,
                result.get('products', [])
            )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@ai_router.get("/conversation/{conversation_id}")
async def get_conversation(conversation_id: str):
    """Get conversation history"""
    conversation = chat_handler.get_conversation(conversation_id)
    if not conversation:
        raise HTTPException(status_code=404, detail="Conversation not found")
    
    return {
        "success": True,
        "conversation": conversation
    }


@ai_router.post("/products/search")
async def search_products(request: ProductRequest):
    """Search for products"""
    try:
        recommendations = ai_service.get_product_recommendations(
            keyword=request.search_query,
            limit=request.limit
        )
        
        return {
            "success": True,
            "query": request.search_query,
            "count": len(recommendations),
            "products": recommendations
        }
        
    except Exception as e:
        logger.error(f"Product search error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching products: {str(e)}"
        )


@ai_router.post("/order/track")
async def track_order(request: OrderRequest):
    """Track order status"""
    try:
        order_info = ai_service.track_order(request.order_id)
        
        if not order_info:
            raise HTTPException(status_code=404, detail="Order not found")
        
        return {
            "success": True,
            "order": order_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Order tracking error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error tracking order: {str(e)}"
        )


@ai_router.get("/health")
async def health_check():
    """Health check endpoint"""
    pinecone_status = ai_service.get_pinecone_status()
    
    return {
        "status": "healthy",
        "service": "E-commerce AI Customer Support",
        "version": "2.0.0",
        "features": [
            "Multilingual support (English/Bengali/Banglish)",
            "Vector search with 768-dim embeddings",
            "Product recommendations",
            "Order tracking"
        ],
        "pinecone": pinecone_status
    }


@ai_router.get("/models")
async def get_available_models():
    """Get available LLM models"""
    try:
        # This would need to be implemented in LLMManager
        return {
            "success": True,
            "models": ["mixtral-8x7b-32768", "llama2-70b-4096", "llama3-70b-8192"]
        }
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {
            "success": False,
            "error": str(e)
        }