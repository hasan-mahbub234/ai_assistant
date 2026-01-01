from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
from .ai import EcommerceAICustomerSupport
import logging

logger = logging.getLogger(__name__)

ai_router = APIRouter()

ai_support = EcommerceAICustomerSupport()

conversation_store = {}

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[int] = None
    conversation_id: Optional[str] = None

class OrderTrackingRequest(BaseModel):
    order_id: str
    user_id: Optional[str] = None

class ProductSuggestionRequest(BaseModel):
    search_query: str
    limit: Optional[int] = 5

class SourceDocument(BaseModel):
    content: str
    metadata: dict

class ChatResponse(BaseModel):
    response: str
    sources: List[SourceDocument]
    language: str
    chat_id: str
    user_type: str  # Added user_type field
    conversation_history: Optional[List[dict]] = None
    success: bool = True

class OrderTrackingResponse(BaseModel):
    order_id: str
    customer_name: str
    status: str
    total_amount: int
    payment_status: str
    address: str
    order_date: str
    products: List[dict]
    success: bool = True

class ProductRecommendation(BaseModel):
    product_id: str
    name: str
    brand: str
    price: int
    average_rating: float
    total_reviews: int
    top_reviews: List[dict]

class ProductSuggestionResponse(BaseModel):
    recommendations: List[ProductRecommendation]
    success: bool = True

@ai_router.get("/conversation/{chat_id}")
async def get_conversation_history(chat_id: str):
    """Retrieve full conversation history by chat_id"""
    if chat_id not in conversation_store:
        return {
            "success": False,
            "error": "Conversation not found",
            "chat_id": chat_id
        }
    
    return {
        "success": True,
        "chat_id": chat_id,
        "messages": conversation_store[chat_id].get("messages", []),
        "user_id": conversation_store[chat_id].get("user_id"),
        "user_type": conversation_store[chat_id].get("user_type"),
        "created_at": conversation_store[chat_id].get("created_at"),
        "product_context": conversation_store[chat_id].get("product_context")  # Return product context
    }

@ai_router.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Chat endpoint with multilingual support and conversation tracking"""
    try:
        logger.info(f"[v0] Chat request: message='{request.message[:50]}...' user_id={request.user_id} conversation_id={request.conversation_id}")
        
        if request.conversation_id and request.conversation_id in conversation_store:
            # Existing conversation
            chat_id = request.conversation_id
            user_type = conversation_store[chat_id].get("user_type")
            logger.info(f"[v0] Continuing existing conversation: {chat_id} ({user_type})")
        else:
            # New conversation
            if request.user_id and request.user_id > 0:
                chat_id = ai_support.generate_chat_id(request.user_id)
                user_type = "authenticated_user"
                logger.info(f"[v0] New authenticated conversation: {chat_id} for user_id={request.user_id}")
            else:
                chat_id = ai_support.generate_chat_id(None)
                user_type = "guest_user"
                logger.info(f"[v0] New guest conversation: {chat_id}")
        
        if chat_id not in conversation_store:
            conversation_store[chat_id] = {
                "user_id": request.user_id,
                "user_type": user_type,
                "messages": [],
                "product_context": {},  # Track mentioned products
                "created_at": str(__import__('datetime').datetime.now()),
                "updated_at": str(__import__('datetime').datetime.now())
            }
        
        # Get user context
        user_context = ai_support.get_user_context(request.user_id)
        
        product_context = conversation_store[chat_id].get("product_context", {})
        
        result = ai_support.get_customer_response(
            user_query=request.message,
            user_context=user_context,
            user_id=request.user_id,
            chat_id=chat_id,
            conversation_history=conversation_store[chat_id]["messages"],
            product_context=product_context
        )
        
        logger.info(f"[v0] AI Response generated. Language: {result.get('language')}")
        
        conversation_store[chat_id]["messages"].append({
            "role": "user",
            "content": request.message,
            "timestamp": str(__import__('datetime').datetime.now())
        })
        
        conversation_store[chat_id]["messages"].append({
            "role": "assistant",
            "content": result["answer"],
            "timestamp": str(__import__('datetime').datetime.now())
        })
        
        if result.get("product_context"):
            conversation_store[chat_id]["product_context"].update(result.get("product_context"))
        
        conversation_store[chat_id]["updated_at"] = str(__import__('datetime').datetime.now())
        
        # Extract source information
        sources = []
        for doc in result["source_documents"]:
            sources.append({
                "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                "metadata": doc.metadata
            })
        
        return ChatResponse(
            response=result["answer"],
            sources=sources,
            language=result["language"],
            chat_id=chat_id,
            user_type=user_type,
            conversation_history=conversation_store[chat_id]["messages"],
            success=True
        )
    
    except Exception as e:
        logger.error(f"[v0] Chat error: {str(e)}")
        import traceback
        traceback.print_exc()
        
        chat_id = request.conversation_id or ai_support.generate_chat_id(request.user_id)
        user_type = "guest_user" if not request.user_id else "authenticated_user"
        
        return ChatResponse(
            response="Sorry, I encountered an error. Please try again.",
            sources=[],
            language="english",
            chat_id=chat_id,
            user_type=user_type,
            success=False
        )

@ai_router.post("/track-order")
async def track_order(request: OrderTrackingRequest):
    """Get order tracking information for a specific order"""
    try:
        print(f"[v0] Order tracking request for order: {request.order_id}")
        
        order_info = ai_support.get_order_tracking_info(request.order_id)
        
        if not order_info:
            return {
                "success": False,
                "error": "Order not found",
                "order_id": request.order_id
            }
        
        return {
            **order_info,
            "success": True
        }
    
    except Exception as e:
        print(f"[v0] Order tracking error: {str(e)}")
        return {
            "success": False,
            "error": f"Error tracking order: {str(e)}",
            "order_id": request.order_id
        }

@ai_router.post("/product-suggestions")
async def get_product_suggestions(request: ProductSuggestionRequest):
    """Get product recommendations with reviews based on search query"""
    try:
        print(f"[v0] Product suggestion request for: {request.search_query}")
        
        recommendations = ai_support.get_product_recommendations_with_reviews(
            keyword=request.search_query,
            limit=request.limit or 5
        )
        
        if not recommendations:
            return {
                "success": False,
                "error": "No products found",
                "query": request.search_query
            }
        
        return {
            "recommendations": recommendations,
            "success": True
        }
    
    except Exception as e:
        print(f"[v0] Product suggestion error: {str(e)}")
        return {
            "success": False,
            "error": f"Error getting suggestions: {str(e)}",
            "query": request.search_query
        }

@ai_router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Multilingual AI Customer Support with Product Search"}

@ai_router.post("/search-product")
async def search_product(request: ChatRequest):
    """Direct product search endpoint"""
    try:
        products = ai_support.search_products_from_db(request.message)
        return {
            "products": products,
            "count": len(products),
            "success": True
        }
    except Exception as e:
        print(f"[v0] Search error: {str(e)}")
        return {
            "products": [],
            "count": 0,
            "error": str(e),
            "success": False
        }
