"""
Document API routes
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, UploadFile, File, Query, HTTPException, status, Form
from fastapi.responses import StreamingResponse, JSONResponse
from sqlalchemy.orm import Session
import io

from src.database import get_db
from src.auth.dependencies import get_current_active_user
from src.auth.models import User
from src.documents.schemas import (
    DocumentUpload, DocumentResponse, DocumentDetailResponse,
    DocumentListResponse, DocumentAnalysisResponse,
)


router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED, operation_id="upload_document")
async def upload_document(
    file: UploadFile = File(...),
    description: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Upload a legal document
    """
    from src.documents.services import document_service
    from src.documents.schemas import DocumentUpload
    
    # Parse tags if provided
    tag_list = []
    if tags:
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
    
    document_metadata = DocumentUpload(
        description=description,
        tags=tag_list
    )
    
    document = await document_service.upload_document(db, file, current_user.id, document_metadata)
    return DocumentResponse.model_validate(document)


@router.get("", response_model=DocumentListResponse, operation_id="list_documents")
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=100),
    status: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    from src.documents.services import document_service
    """
    Get all documents for the current user
    """
    documents, total = document_service.get_user_documents(
        db, current_user.id, skip, limit, status
    )
    
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=total,
        page=skip // limit + 1 if limit > 0 else 1,
        page_size=limit,
        total_pages=(total + limit - 1) // limit if limit > 0 else 1,
    )


@router.get("/{document_id}", response_model=DocumentDetailResponse, operation_id="get_document")
async def get_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Get a specific document
    """
    from src.documents.services import document_service
    document = document_service.get_document(db, document_id, current_user.id)
    return DocumentDetailResponse.model_validate(document)


@router.delete("/{document_id}", operation_id="delete_document")
async def delete_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Delete a document
    """
    from src.documents.services import document_service
    success = await document_service.delete_document(db, document_id, current_user.id)
    return {"message": "Document deleted successfully"}


@router.get("/{document_id}/download", operation_id="download_document")
async def download_document(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Download a document
    """
    from src.documents.services import document_service
    content, filename = await document_service.download_document(db, document_id, current_user.id)
    
    return StreamingResponse(
        io.BytesIO(content),
        media_type="application/octet-stream",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Content-Length": str(len(content)),
        }
    )


@router.get("/{document_id}/analysis", response_model=DocumentAnalysisResponse, operation_id="get_document_analysis")
async def get_document_analysis(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Get AI analysis of a document
    """
    from src.documents.services import document_service
    analysis = document_service.get_document_analysis(db, document_id, current_user.id)
    
    # Use the helper method to create response
    return DocumentAnalysisResponse.from_analysis(analysis)


@router.get("/{document_id}/text", operation_id="get_document_text")
async def get_document_text(
    document_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Get extracted text from document
    """
    from src.documents.services import document_service
    document = document_service.get_document(db, document_id, current_user.id)
    
    if not document.extracted_text:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document text not available",
        )
    
    return JSONResponse(
        content={
            "text": document.extracted_text,
            "document_id": document.id,
            "filename": document.original_filename,
        }
    )