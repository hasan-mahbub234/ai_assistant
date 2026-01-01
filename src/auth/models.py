"""
Authentication database models
"""
import enum
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, 
    ForeignKey, Enum, Index, Text
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import ENUM

from src.database import Base


class UserRole(str, enum.Enum):
    """User role enumeration"""
    SUPER_ADMIN = "super_admin"
    ADMIN = "admin"
    LAWYER = "lawyer"
    PARALEGAL = "paralegal"
    CLIENT = "client"


class UserStatus(str, enum.Enum):
    """User status enumeration"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"
    PENDING = "pending"


class User(Base):
    """
    User model for authentication and authorization
    """
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=False)
    role = Column(ENUM(UserRole, name="user_role"), default=UserRole.CLIENT, nullable=False)
    status = Column(ENUM(UserStatus, name="user_status"), default=UserStatus.PENDING, nullable=False)
    
    # Profile fields
    phone = Column(String(20), nullable=True)
    avatar_url = Column(String(500), nullable=True)
    bio = Column(Text, nullable=True)
    
    # Security fields
    is_email_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime(timezone=True), nullable=True)
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    refresh_tokens = relationship("RefreshToken", back_populates="user", cascade="all, delete-orphan")
    api_keys = relationship("ApiKey", back_populates="user", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('ix_users_email_status', 'email', 'status'),
        Index('ix_users_role_status', 'role', 'status'),
    )
    
    def __repr__(self):
        return f"<User {self.email}>"


class RefreshToken(Base):
    """
    Refresh token model for JWT rotation
    """
    __tablename__ = "refresh_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(500), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_revoked = Column(Boolean, default=False)
    revoked_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Device information
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="refresh_tokens")
    
    __table_args__ = (
        Index('ix_refresh_tokens_user_id', 'user_id'),
        Index('ix_refresh_tokens_token', 'token'),
    )


class ApiKey(Base):
    """
    API key model for programmatic access
    """
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    key_hash = Column(String(64), unique=True, nullable=False)  # SHA256 hash
    last_used_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Scopes (comma-separated)
    scopes = Column(Text, default="read", nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    __table_args__ = (
        Index('ix_api_keys_user_id', 'user_id'),
        Index('ix_api_keys_key_hash', 'key_hash'),
    )


class PasswordResetToken(Base):
    """
    Password reset token model
    """
    __tablename__ = "password_reset_tokens"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    token = Column(String(64), unique=True, nullable=False)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    is_used = Column(Boolean, default=False)
    used_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    __table_args__ = (
        Index('ix_password_reset_tokens_token', 'token'),
        Index('ix_password_reset_tokens_user_id', 'user_id'),
    )