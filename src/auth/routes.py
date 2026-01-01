"""
Authentication API routes
"""
from typing import List
from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer
from sqlalchemy.orm import Session
from src.core.config import settings
from src.database import get_db
from src.auth.dependencies import (
    get_current_user, get_current_active_user, get_user_response,
    get_current_admin_user,
)
from src.auth.models import User
from src.auth.schemas import (
    UserCreate, UserLogin, TokenResponse,
    PasswordResetRequest, PasswordResetConfirm,
    UserUpdate, ChangePassword, RefreshTokenRequest,
    ApiKeyCreate, ApiKeyResponse, ApiKeyListResponse, UserResponse
)


router = APIRouter(prefix="/auth", tags=["auth"])
security = HTTPBearer()


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED, operation_id="register")
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db),
):
    """
    Register a new user
    """
    from src.auth.services import auth_service
    user, access_token, refresh_token = auth_service.register_user(db, user_data)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=get_user_response(user),
    )


@router.post("/login", response_model=TokenResponse, operation_id="login")
async def login(
    login_data: UserLogin,
    request: Request,
    db: Session = Depends(get_db),
):
    """
    Login user
    """
    from src.auth.services import auth_service
    user_agent = request.headers.get("user-agent")
    ip_address = request.client.host if request.client else None
    
    user, access_token, refresh_token = auth_service.login_user(
        db, login_data, user_agent, ip_address
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=get_user_response(user),
    )


@router.post("/refresh", response_model=TokenResponse, operation_id="refresh_token")
async def refresh_token(
    token_data: RefreshTokenRequest,
    db: Session = Depends(get_db),
):
    """
    Refresh access token
    """
    from src.auth.services import auth_service
    access_token, refresh_token = auth_service.refresh_access_token(db, token_data.refresh_token)
    
    # Get user from new token
    from src.core.security import decode_token
    payload = decode_token(access_token)
    user_id = payload.get("sub")
    
    user = db.query(User).filter(User.id == int(user_id)).first()
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=get_user_response(user),
    )


@router.post("/logout", operation_id="logout")
async def logout(
    token_data: RefreshTokenRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Logout user
    """
    from src.auth.services import auth_service
    auth_service.logout_user(db, token_data.refresh_token, current_user.id)
    return {"message": "Successfully logged out"}


@router.post("/password/reset-request", operation_id="password_reset_request")
async def password_reset_request(
    reset_data: PasswordResetRequest,
    db: Session = Depends(get_db),
):
    """
    Request password reset
    """
    from src.auth.services import auth_service
    message = auth_service.request_password_reset(db, reset_data.email)
    return {"message": message}


@router.post("/password/reset", operation_id="password_reset")
async def password_reset(
    reset_data: PasswordResetConfirm,
    db: Session = Depends(get_db),
):
    """
    Reset password using token
    """
    from src.auth.services import auth_service
    success = auth_service.reset_password(db, reset_data.token, reset_data.new_password)
    return {"message": "Password reset successful"}


@router.get("/me", response_model=UserResponse, operation_id="get_current_user_profile")
async def get_current_user_profile(
    current_user: User = Depends(get_current_active_user),
):
    """
    Get current user profile
    """
    return get_user_response(current_user)


@router.put("/me", response_model=UserResponse, operation_id="update_user_profile")
async def update_user_profile(
    update_data: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Update user profile
    """
    from src.auth.services import auth_service
    user = auth_service.update_user_profile(db, current_user.id, update_data)
    return get_user_response(user)


@router.post("/me/change-password", operation_id="change_password")
async def change_password(
    change_data: ChangePassword,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Change user password
    """
    from src.auth.services import auth_service
    auth_service.change_password(db, current_user.id, change_data)
    return {"message": "Password changed successfully"}


@router.post("/me/api-keys", response_model=ApiKeyResponse, operation_id="create_api_key")
async def create_api_key(
    api_key_data: ApiKeyCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Create API key
    """
    from src.auth.services import auth_service
    api_key, api_key_record = auth_service.create_api_key(db, current_user.id, api_key_data)
    
    return ApiKeyResponse(
        id=api_key_record.id,
        name=api_key_record.name,
        key=api_key,  # Show only once
        expires_at=api_key_record.expires_at,
        created_at=api_key_record.created_at,
        last_used_at=api_key_record.last_used_at,
    )


@router.get("/me/api-keys", response_model=ApiKeyListResponse, operation_id="get_api_keys")
async def get_api_keys(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Get user API keys
    """
    from src.auth.services import auth_service
    api_keys = auth_service.get_user_api_keys(db, current_user.id)
    
    return ApiKeyListResponse(
        keys=[
            ApiKeyResponse(
                id=key.id,
                name=key.name,
                key="***",  # Never show actual key
                expires_at=key.expires_at,
                created_at=key.created_at,
                last_used_at=key.last_used_at,
            )
            for key in api_keys
        ],
        total=len(api_keys),
    )


@router.delete("/me/api-keys/{api_key_id}", operation_id="revoke_api_key")
async def revoke_api_key(
    api_key_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """
    Revoke API key
    """
    from src.auth.services import auth_service
    auth_service.revoke_api_key(db, current_user.id, api_key_id)
    return {"message": "API key revoked successfully"}


@router.get("/admin/users", response_model=List[UserResponse], operation_id="get_all_users")
async def get_all_users(
    current_user: User = Depends(get_current_admin_user),
    db: Session = Depends(get_db),
    skip: int = 0,
    limit: int = 100,
):
    """
    Get all users (admin only)
    """
    users = db.query(User).offset(skip).limit(limit).all()
    return [get_user_response(user) for user in users]