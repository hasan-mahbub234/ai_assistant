"""
Pydantic schemas for authentication
"""
from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, validator
import re

from src.auth.models import UserRole, UserStatus


# Request Schemas
class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)
    full_name: str = Field(..., min_length=2, max_length=100)
    role: UserRole = UserRole.CLIENT
    phone: Optional[str] = None
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^\+?1?\d{9,15}$', v):
            raise ValueError('Invalid phone number format')
        return v


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str
    remember_me: bool = False


class PasswordResetRequest(BaseModel):
    """Schema for password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation"""
    token: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v


class UserUpdate(BaseModel):
    """Schema for user profile update"""
    full_name: Optional[str] = Field(None, min_length=2, max_length=100)
    phone: Optional[str] = None
    bio: Optional[str] = Field(None, max_length=500)
    
    @validator('phone')
    def validate_phone(cls, v):
        if v and not re.match(r'^\+?1?\d{9,15}$', v):
            raise ValueError('Invalid phone number format')
        return v


class ChangePassword(BaseModel):
    """Schema for password change"""
    current_password: str
    new_password: str = Field(..., min_length=8, max_length=100)
    
    @validator('new_password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        if not re.search(r'[A-Z]', v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not re.search(r'[a-z]', v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not re.search(r'\d', v):
            raise ValueError('Password must contain at least one number')
        return v


# Response Schemas
class UserResponse(BaseModel):
    """Schema for user response"""
    id: int
    email: EmailStr
    full_name: str
    role: UserRole
    status: UserStatus
    phone: Optional[str]
    avatar_url: Optional[str]
    bio: Optional[str]
    is_email_verified: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """Schema for token response"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """Schema for refresh token request"""
    refresh_token: str


class ApiKeyCreate(BaseModel):
    """Schema for API key creation"""
    name: str = Field(..., min_length=1, max_length=100)
    expires_in_days: Optional[int] = Field(None, ge=1, le=365)


class ApiKeyResponse(BaseModel):
    """Schema for API key response"""
    id: int
    name: str
    key: str  # Only shown on creation
    expires_at: Optional[datetime]
    created_at: datetime
    last_used_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class ApiKeyListResponse(BaseModel):
    """Schema for API key list response"""
    keys: List[ApiKeyResponse]
    total: int