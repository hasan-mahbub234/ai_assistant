from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from src.ai.routes import ai_router
from src.config import config

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting E-commerce AI Service...")
    
    try:
        # Import and initialize AI service (will be initialized on first use)
        from src.ai.ai_service import AIService
        ai_service = AIService()  # This triggers initialization
        
        # Check Pinecone status
        pinecone_status = ai_service.get_pinecone_status()
        logger.info(f"📊 Pinecone Status: {pinecone_status}")
        
        if pinecone_status.get('status') == 'connected':
            vector_count = pinecone_status.get('stats', {}).get('total_vector_count', 0)
            logger.info(f"✅ Pinecone connected with {vector_count} vectors")
        else:
            logger.warning(f"⚠️ Pinecone: {pinecone_status.get('message', 'Unknown status')}")
        
        logger.info("✅ AI Service ready to accept requests")
        
    except Exception as e:
        logger.error(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
    
    yield
    
    logger.info("🛑 Shutting down AI Service...")


# Create FastAPI app
app = FastAPI(
    title="E-commerce AI Customer Support API",
    description="Multilingual AI Customer Support with Product Search & Order Tracking",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include AI router
app.include_router(ai_router, prefix="/api/v1/ai", tags=["AI"])


@app.get("/")
async def root():
    return {
        "message": "E-commerce AI Customer Support API",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "chat": "/api/v1/ai/chat",
            "product_search": "/api/v1/ai/products/search",
            "order_tracking": "/api/v1/ai/order/track",
            "health": "/api/v1/ai/health"
        }
    }


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "AI Customer Support"}


# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.PORT,
        log_level=config.LOG_LEVEL.lower()
    )