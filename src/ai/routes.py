"""
AI API routes
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from src.database import get_db
from src.auth.dependencies import get_current_active_user
from src.auth.models import User
from src.documents.models import ChatHistory, DocumentStatus
from src.documents.services import document_service
from src.ai.services import ai_processor
from src.ai.schemas import (
    ChatRequest,
    ChatResponse,
    ChatHistoryResponse,
    DocumentAnalysisRequest,
)

router = APIRouter(prefix="/ai", tags=["ai"])


# =========================================================
# Chat with document (RAG)
# =========================================================

@router.post("/chat", response_model=ChatResponse, operation_id="chat")
async def chat_with_document(
    request: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Verify document ownership
    document = document_service.get_document(
        db, request.document_id, current_user.id
    )

    # FIX: Compare with DocumentStatus Enum value, not string
    if document.status != DocumentStatus.PROCESSED:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Document is not ready for chat. Current status: {document.status.value}. Please wait for processing to complete.",
        )

    # Run AI chat
    result = await ai_processor.chat_with_document(
        document_id=request.document_id,
        question=request.question,
        temperature=request.temperature,
    )

    # Persist chat
    chat_record = ChatHistory(
        document_id=request.document_id,
        user_id=current_user.id,
        question=request.question,
        answer=result["answer"],
        model_used=ai_processor.chat_model,
        temperature=request.temperature,
        context_chunks=result.get("sources", []),
        response_time_ms=int(result["response_time"] * 1000),
    )

    db.add(chat_record)
    db.commit()
    db.refresh(chat_record)

    return ChatResponse(
        answer=result["answer"],
        sources=result.get("sources", []),
        response_time_ms=int(result["response_time"] * 1000),
    )


# =========================================================
# Chat history
# =========================================================

@router.get(
    "/chat-history/{document_id}",
    response_model=ChatHistoryResponse,
    operation_id="chat_history",
)
async def get_chat_history(
    document_id: int,
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    # Verify ownership
    document_service.get_document(db, document_id, current_user.id)

    query = db.query(ChatHistory).filter(
        ChatHistory.document_id == document_id,
        ChatHistory.user_id == current_user.id,
    )

    total = query.count()

    records = (
        query.order_by(ChatHistory.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return ChatHistoryResponse(
        history=records,
        total=total,
        page=(skip // limit) + 1 if limit else 1,
        page_size=limit,
    )


# =========================================================
# Analyze raw text (no storage)
# =========================================================

@router.post("/analyze-text", operation_id="analyze_text")
async def analyze_text(
    request: DocumentAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
):
    try:
        result = await ai_processor.analyze_document(request.text)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AI analysis failed: {str(e)}",
        ) from e


# =========================================================
# Extract clauses (rule-based) - TEMPORARILY DISABLED
# =========================================================

@router.get("/clauses/{document_id}", operation_id="extract_clauses")
async def extract_clauses(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = document_service.get_document(
        db, document_id, current_user.id
    )

    if not document.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text not available",
        )

    # Return empty for now since extract_clauses method doesn't exist
    return {"clauses": []}


# =========================================================
# Summarize document
# =========================================================

@router.post("/summarize/{document_id}", operation_id="summarize_document")
async def summarize_document(
    document_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    document = document_service.get_document(
        db, document_id, current_user.id
    )

    if not document.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document text not available",
        )

    if document.summary:
        return {"summary": document.summary}

    try:
        result = await ai_processor.analyze_document(
            document.extracted_text
        )

        summary = result.get("summary", "")
        if summary and summary != "Analysis unavailable.":
            document.summary = summary
            db.commit()

        return {"summary": summary}

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate summary: {str(e)}",
        ) from e


# =========================================================
# Test AI configuration
# =========================================================

@router.get("/test-config", operation_id="test_ai_configuration")
async def test_ai_configuration():
    """Test if AI services are properly configured"""
    config_status = {
        "groq_configured": ai_processor.client is not None,
        "pinecone_configured": ai_processor.index is not None,
        "embedding_model": "sentence-transformers/all-mpnet-base-v2",
        "groq_model": getattr(ai_processor, 'chat_model', 'Not configured')
    }
    
    # Test with a simple query
    test_text = "This is a test document for analysis. It contains sample legal text for testing purposes."
    try:
        result = await ai_processor.analyze_document(test_text)
        config_status["ai_test_passed"] = True
        config_status["test_result"] = {
            "summary": result.get("summary", ""),
            "has_error": "error" in result
        }
        if "error" in result:
            config_status["error"] = result["error"]
    except Exception as e:
        config_status["ai_test_passed"] = False
        config_status["error"] = str(e)
    
    return config_status


# =========================================================
# Health check for AI services
# =========================================================

@router.get("/health", operation_id="ai_health_check")
async def ai_health_check():
    """Check health of AI services"""
    health_status = {
        "status": "healthy",
        "services": {
            "groq": ai_processor.client is not None,
            "pinecone": ai_processor.index is not None,
            "embeddings": hasattr(ai_processor, 'embedder'),
        },
        "model": getattr(ai_processor, 'chat_model', 'Not configured'),
        "recommendations": []
    }
    
    if not health_status["services"]["groq"]:
        health_status["recommendations"].append("Groq API key not configured")
    
    if not health_status["services"]["pinecone"]:
        health_status["recommendations"].append("Pinecone not configured")
    
    if health_status["recommendations"]:
        health_status["status"] = "degraded"
    
    return health_status