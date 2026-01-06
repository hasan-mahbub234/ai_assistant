import os
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import time  # Add this import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import and initialize AI service lazily
ai_service = None
chat_handler = None

def get_ai_service():
    """Lazy initialization of AI service"""
    global ai_service
    if ai_service is None:
        from src.ai.ai_service import AIService
        ai_service = AIService()
    return ai_service

def get_chat_handler():
    """Lazy initialization of chat handler"""
    global chat_handler
    if chat_handler is None:
        from src.ai.chat_handler import ChatHandler
        chat_handler = ChatHandler()
    return chat_handler

app = FastAPI(
    title="E-commerce AI Customer Support API",
    description="Multilingual AI Customer Support with Product Search & Order Tracking",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple request/response models
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

# Initialize services on startup
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting up AI service...")
    try:
        service = get_ai_service()
        status = service.get_pinecone_status()
        logger.info(f"Pinecone status: {status}")
        logger.info("✅ AI Service initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize AI service: {e}")

@app.post("/api/v1/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint with AI"""
    try:
        logger.info(f"Chat request: {request.message[:100]}")
        
        # Get AI service
        ai_service = get_ai_service()
        chat_handler = get_chat_handler()
        
        # Get AI response
        result = ai_service.get_response(
            user_query=request.message,
            user_id=request.user_id,
            conversation_id=request.conversation_id
        )
        
        # Store in chat handler if conversation_id provided
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
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        return ChatResponse(
            answer="I apologize, but I'm having trouble processing your request right now. Please try again.",
            language="english",
            conversation_id=request.conversation_id or "error",
            success=False,
            products=[]
        )

@app.get("/api/v1/ai/health")
async def health_check():
    """Health check endpoint"""
    try:
        ai_service = get_ai_service()
        pinecone_status = ai_service.get_pinecone_status()
        
        return {
            "status": "healthy",
            "service": "E-commerce AI Customer Support",
            "version": "2.0.0",
            "pinecone": pinecone_status
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {
        "message": "E-commerce AI Customer Support API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/v1/ai/chat",
            "health": "/api/v1/ai/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, reload=True)