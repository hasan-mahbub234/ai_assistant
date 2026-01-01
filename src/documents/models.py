"""
Document management database models
"""
import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, DateTime, ForeignKey,
    Boolean, Enum, Index, Float, JSON
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import UUID

from src.database import Base


class DocumentStatus(enum.Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class Document(Base):
    """
    Legal document model
    """
    __tablename__ = "documents"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # File information
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)  # in bytes
    mime_type = Column(String(100), nullable=True)
    
    # Content
    extracted_text = Column(Text, nullable=True)
    summary = Column(Text, nullable=True)
    document_metadata = Column(JSON, nullable=True, default=dict)
    
    # Processing status
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING, nullable=False)
    processing_started_at = Column(DateTime(timezone=True), nullable=True)
    processing_completed_at = Column(DateTime(timezone=True), nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # AI Analysis results (stored as JSON)
    ai_analysis = Column(JSON, nullable=True, default=dict)
    
    # Security
    is_encrypted = Column(Boolean, default=False)
    encryption_key = Column(String(255), nullable=True)
    
    # Ownership
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", backref="documents")
    embeddings = relationship("DocumentEmbedding", back_populates="document", cascade="all, delete-orphan")
    chat_history = relationship("ChatHistory", back_populates="document", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_documents_user_id_status', 'user_id', 'status'),
        Index('ix_documents_created_at', 'created_at'),
        Index('ix_documents_original_filename', 'original_filename'),
    )
    
    def __repr__(self):
        return f"<Document {self.original_filename}>"


class DocumentEmbedding(Base):
    """
    Document embeddings for vector search
    """
    __tablename__ = "document_embeddings"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    
    # Pinecone reference
    pinecone_id = Column(String(255), nullable=False)
    
    # Embedding metadata
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    chunk_start = Column(Integer, nullable=False)  # Character position
    chunk_end = Column(Integer, nullable=False)    # Character position
    
    # Model information
    embedding_model = Column(String(100), nullable=False)
    embedding_dimension = Column(Integer, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="embeddings")
    
    __table_args__ = (
        Index('ix_document_embeddings_document_id', 'document_id'),
        Index('ix_document_embeddings_pinecone_id', 'pinecone_id'),
        Index('ix_document_embeddings_chunk_index', 'chunk_index'),
    )


class ChatHistory(Base):
    """
    Chat history with documents
    """
    __tablename__ = "chat_history"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Conversation
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    
    # AI metadata
    model_used = Column(String(100), nullable=True)
    temperature = Column(Float, nullable=True)
    context_chunks = Column(JSON, nullable=True)  # Store which chunks were used
    
    # Performance metrics
    response_time_ms = Column(Integer, nullable=True)
    token_count = Column(Integer, nullable=True)
    
    # User feedback
    is_helpful = Column(Boolean, nullable=True)
    user_feedback = Column(Text, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="chat_history")
    user = relationship("User", backref="chat_history")
    
    __table_args__ = (
        Index('ix_chat_history_document_id_user_id', 'document_id', 'user_id'),
        Index('ix_chat_history_created_at', 'created_at'),
    )


class DocumentShare(Base):
    """
    Document sharing permissions
    """
    __tablename__ = "document_shares"
    
    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    shared_by_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    shared_with_user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    
    # Permissions
    can_view = Column(Boolean, default=True)
    can_edit = Column(Boolean, default=False)
    can_delete = Column(Boolean, default=False)
    can_share = Column(Boolean, default=False)
    
    # Expiry
    expires_at = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    document = relationship("Document")
    shared_by = relationship("User", foreign_keys=[shared_by_user_id])
    shared_with = relationship("User", foreign_keys=[shared_with_user_id])
    
    __table_args__ = (
        Index('ix_document_shares_document_id', 'document_id'),
        Index('ix_document_shares_shared_with_user_id', 'shared_with_user_id'),
        Index('ix_document_shares_shared_by_user_id', 'shared_by_user_id'),
        Index('ix_document_shares_expires_at', 'expires_at'),
    )