# In __init__.py, replace the current content with:

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import gc

from src.ai.routes import ai_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

version = "v1"

ai_support_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown events for AI service only."""
    # Memory optimization
    gc.collect()
    os.environ['TRANSFORMERS_CACHE'] = '/tmp'
    os.environ['HF_HOME'] = '/tmp'
    
    logger.info("🚀 AI Service is starting...")
    
    global ai_support_instance
    
    # Initialize AI system (ONLY ONCE via singleton)
    try:
        from src.ai.ai import EcommerceAICustomerSupport
        ai_support_instance = EcommerceAICustomerSupport()
        logger.info("✅ AI system initialized successfully (singleton pattern)")
        
        # Check Pinecone status
        pinecone_status = ai_support_instance.check_pinecone_status()
        logger.info(f"📊 Pinecone Status: {pinecone_status}")
        
    except Exception as e:
        logger.error(f"❌ AI system initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        # Don't raise, let the app start without AI if needed
    
    yield
    
    logger.info("🛑 AI Service is shutting down...")

def create_app() -> FastAPI:
    """Factory function to create and configure the AI-only FastAPI application."""
    app = FastAPI(
        title="E-commerce AI Service",
        description="AI Customer Support & Product Recommendation API",
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc"
    )

    # Configure CORS for Vercel
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global exception handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"message": "Internal server error"},
        )

    # Include ONLY AI router
    app.include_router(
        ai_router,
        prefix=f"/api/{version}/ai",
        tags=["ai"]
    )

    # Health check endpoint
    @app.get("/health", tags=["monitoring"])
    async def health_check():
        return {
            "status": "healthy", 
            "service": "AI Customer Support",
            "deployment": "railway"
        }
    
    # Root endpoint
    @app.get("/")
    async def root():
        return {
            "message": "E-commerce AI Customer Support API",
            "version": version,
            "docs": "/docs"
        }

    return app

# Create the application instance
app = create_app()

# For Railway deployment
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        workers=1,
        log_level="info"
    )