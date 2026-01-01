"""
Pydantic schemas for AI services
"""
from datetime import datetime
from typing import List, Optional, Any, Dict
from pydantic import BaseModel, Field, ConfigDict


class ChatRequest(BaseModel):
    """Schema for chat request"""
    document_id: int
    question: str = Field(..., min_length=1, max_length=1000)
    temperature: float = Field(0.1, ge=0, le=1)


class ChatSource(BaseModel):
    """Schema for chat source citation"""
    chunk_index: int
    score: float
    
    model_config = ConfigDict(from_attributes=True)


class ChatResponse(BaseModel):
    """Schema for chat response"""
    answer: str
    sources: List[ChatSource]
    response_time_ms: int
    
    model_config = ConfigDict(from_attributes=True)


class ChatHistoryItem(BaseModel):
    """Schema for chat history item"""
    id: int
    question: str
    answer: str
    model_used: Optional[str] = None
    response_time_ms: Optional[int] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


class ChatHistoryResponse(BaseModel):
    """Schema for chat history response"""
    history: List[ChatHistoryItem]
    total: int
    page: int
    page_size: int


class DocumentAnalysisRequest(BaseModel):
    """Schema for document analysis request"""
    text: str = Field(..., min_length=10, max_length=10000)


class DocumentAnalysisResponse(BaseModel):
    """Schema for document analysis response"""
    summary: str
    key_points: List[str]
    clauses: List[str]
    risks: List[str]
    recommendations: List[str]
    processing_time: float
    error: Optional[str] = None