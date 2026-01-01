"""
Pydantic schemas for document management
"""
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum


# Request Schemas
class DocumentUpload(BaseModel):
    """Schema for document upload"""
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list)


class DocumentUpdate(BaseModel):
    """Schema for document update"""
    description: Optional[str] = Field(None, max_length=500)
    tags: List[str] = Field(default_factory=list)


class DocumentShareCreate(BaseModel):
    """Schema for document sharing"""
    shared_with_user_id: int
    can_view: bool = True
    can_edit: bool = False
    can_delete: bool = False
    can_share: bool = False
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


# Response Schemas
class DocumentResponse(BaseModel):
    """Schema for document response"""
    id: int
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str
    summary: Optional[str] = None
    document_metadata: Dict[str, Any] = Field(default_factory=dict)
    ai_analysis: Optional[Dict[str, Any]] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)
    
    @validator('status', pre=True)
    def convert_status(cls, v):
        if hasattr(v, 'value'):
            return v.value
        return v


class DocumentDetailResponse(DocumentResponse):
    """Schema for detailed document response"""
    extracted_text: Optional[str] = None
    processing_error: Optional[str] = None
    processing_started_at: Optional[datetime] = None
    processing_completed_at: Optional[datetime] = None


class DocumentListResponse(BaseModel):
    """Schema for document list response"""
    documents: List[DocumentResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class DocumentShareResponse(BaseModel):
    """Schema for document share response"""
    id: int
    document_id: int
    shared_with_user_id: int
    shared_by_user_id: int
    can_view: bool
    can_edit: bool
    can_delete: bool
    can_share: bool
    expires_at: Optional[datetime] = None
    created_at: datetime
    
    model_config = ConfigDict(from_attributes=True)


# AI Analysis Schemas
class ClauseType(str, Enum):
    """Legal clause types"""
    LIABILITY = "liability"
    TERMINATION = "termination"
    PAYMENT = "payment"
    CONFIDENTIALITY = "confidentiality"
    INDEMNIFICATION = "indemnification"
    GOVERNING_LAW = "governing_law"
    FORCE_MAJEURE = "force_majeure"
    WARRANTY = "warranty"
    INTELLECTUAL_PROPERTY = "intellectual_property"
    OTHER = "other"


class RiskLevel(str, Enum):
    """Risk levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ExtractedClause(BaseModel):
    """Schema for extracted clause"""
    type: ClauseType
    text: str
    page: Optional[int] = None
    confidence: float = Field(..., ge=0, le=1)


class IdentifiedRisk(BaseModel):
    """Schema for identified risk"""
    description: str
    level: RiskLevel
    clause_text: str
    recommendation: str
    confidence: float = Field(..., ge=0, le=1)


# Helper function to validate analysis data
def validate_analysis_data(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize analysis data"""
    validated = analysis.copy()
    
    # Ensure identified_clauses is a list of dicts
    if 'identified_clauses' in validated:
        clauses = validated['identified_clauses']
        if isinstance(clauses, list):
            validated['identified_clauses'] = [
                clause if isinstance(clause, dict) else {'text': str(clause), 'type': 'OTHER', 'confidence': 0.8}
                for clause in clauses
            ]
    
    # Ensure risks is a list of dicts
    if 'risks' in validated:
        risks = validated['risks']
        if isinstance(risks, list):
            validated['risks'] = [
                risk if isinstance(risk, dict) else {
                    'description': str(risk),
                    'level': 'MEDIUM',
                    'clause_text': '',
                    'recommendation': 'Review this risk carefully',
                    'confidence': 0.7
                }
                for risk in risks
            ]
    
    # Ensure key_points is a list of strings
    if 'key_points' in validated:
        key_points = validated['key_points']
        if isinstance(key_points, list):
            validated['key_points'] = [str(point) for point in key_points]
    
    # Ensure recommendations is a list of strings
    if 'recommendations' in validated:
        recommendations = validated['recommendations']
        if isinstance(recommendations, list):
            validated['recommendations'] = [str(rec) for rec in recommendations]
        else:
            validated['recommendations'] = []
    
    return validated


class DocumentAnalysisResponse(BaseModel):
    """Schema for document analysis response"""
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    identified_clauses: List[Dict[str, Any]] = Field(default_factory=list)
    risks: List[Dict[str, Any]] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    processing_time: float = 0.0
    
    @classmethod
    def from_analysis(cls, analysis: Dict[str, Any]) -> 'DocumentAnalysisResponse':
        """Create response from raw analysis data"""
        validated = validate_analysis_data(analysis)
        return cls(**validated)