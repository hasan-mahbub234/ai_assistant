"""
Authentication service layer
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException, status

from src.core.config import settings
from src.core.security import (
    verify_password, get_password_hash, create_access_token,
    create_refresh_token, decode_token, hash_api_key,
    generate_api_key, generate_password_reset_token,
)
from src.auth.models import User, RefreshToken, ApiKey, PasswordResetToken
from src.auth.schemas import (
    UserCreate, UserLogin, PasswordResetRequest,
    PasswordResetConfirm, UserUpdate, ChangePassword,
    ApiKeyCreate,
)

logger = logging.getLogger(__name__)


class AuthService:
    """Authentication service"""
    
    @staticmethod
    def register_user(db: Session, user_data: UserCreate) -> Tuple[User, str, str]:
        """
        Register a new user
        """
        # Check if user already exists
        existing_user = db.query(User).filter(User.email == user_data.email).first()
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered",
            )
        
        # Hash password
        hashed_password = get_password_hash(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            full_name=user_data.full_name,
            role=user_data.role,
            phone=user_data.phone,
            status="pending",  # Requires email verification
        )
        
        db.add(user)
        db.commit()
        db.refresh(user)
        
        logger.info(f"User registered: {user.email}")
        
        # Generate tokens
        access_token = create_access_token({"sub": str(user.id)})
        refresh_token = create_refresh_token({"sub": str(user.id)})
        
        # Save refresh token
        refresh_token_record = RefreshToken(
            user_id=user.id,
            token=refresh_token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
        )
        db.add(refresh_token_record)
        db.commit()
        
        return user, access_token, refresh_token
    
    @staticmethod
    def login_user(
        db: Session,
        login_data: UserLogin,
        user_agent: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Tuple[User, str, str]:
        """
        Authenticate user and return tokens
        """
        # Find user
        user = db.query(User).filter(User.email == login_data.email).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked. Please try again later.",
            )
        
        # Verify password
        if not verify_password(login_data.password, user.hashed_password):
            # Increment failed attempts
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                logger.warning(f"Account locked: {user.email}")
            
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(timezone.utc)
        db.commit()
        
        # Generate tokens
        access_token = create_access_token({"sub": str(user.id)})
        refresh_token = create_refresh_token({"sub": str(user.id)})
        
        # Save refresh token with device info
        refresh_token_record = RefreshToken(
            user_id=user.id,
            token=refresh_token,
            expires_at=datetime.now(timezone.utc) + timedelta(
                days=30 if login_data.remember_me else settings.REFRESH_TOKEN_EXPIRE_DAYS
            ),
            user_agent=user_agent,
            ip_address=ip_address,
        )
        db.add(refresh_token_record)
        db.commit()
        
        logger.info(f"User logged in: {user.email}")
        return user, access_token, refresh_token
    
    @staticmethod
    def refresh_access_token(db: Session, refresh_token: str) -> Tuple[str, str]:
        """
        Refresh access token using refresh token
        """
        # Verify refresh token
        payload = decode_token(refresh_token)
        if not payload or payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
            )
        
        # Check if token exists in database
        token_record = db.query(RefreshToken).filter(
            RefreshToken.token == refresh_token,
            RefreshToken.is_revoked == False,
            RefreshToken.expires_at > datetime.now(timezone.utc),
        ).first()
        
        if not token_record:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token not found or expired",
            )
        
        # Generate new tokens
        user_id = payload.get("sub")
        new_access_token = create_access_token({"sub": user_id})
        new_refresh_token = create_refresh_token({"sub": user_id})
        
        # Revoke old refresh token
        token_record.is_revoked = True
        token_record.revoked_at = datetime.now(timezone.utc)
        
        # Save new refresh token
        new_token_record = RefreshToken(
            user_id=token_record.user_id,
            token=new_refresh_token,
            expires_at=datetime.now(timezone.utc) + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS),
            user_agent=token_record.user_agent,
            ip_address=token_record.ip_address,
        )
        db.add(new_token_record)
        db.commit()
        
        return new_access_token, new_refresh_token
    
    @staticmethod
    def logout_user(db: Session, refresh_token: str, user_id: int) -> None:
        """
        Logout user by revoking refresh token
        """
        token_record = db.query(RefreshToken).filter(
            RefreshToken.token == refresh_token,
            RefreshToken.user_id == user_id,
            RefreshToken.is_revoked == False,
        ).first()
        
        if token_record:
            token_record.is_revoked = True
            token_record.revoked_at = datetime.now(timezone.utc)
            db.commit()
            logger.info(f"User logged out: {user_id}")
    
    @staticmethod
    def request_password_reset(db: Session, email: str) -> str:
        """
        Request password reset
        """
        user = db.query(User).filter(User.email == email).first()
        if not user:
            # Don't reveal that user doesn't exist
            return "If the email exists, a reset link has been sent."
        
        # Generate reset token
        token = generate_password_reset_token()
        
        # Save token
        reset_token = PasswordResetToken(
            user_id=user.id,
            token=token,
            expires_at=datetime.now(timezone.utc) + timedelta(minutes=settings.PASSWORD_RESET_TOKEN_EXPIRE_MINUTES),
        )
        db.add(reset_token)
        db.commit()
        
        # In production, send email here
        logger.info(f"Password reset requested for: {email}, token: {token}")
        
        return "If the email exists, a reset link has been sent."
    
    @staticmethod
    def reset_password(db: Session, token: str, new_password: str) -> bool:
        """
        Reset password using token
        """
        reset_token = db.query(PasswordResetToken).filter(
            PasswordResetToken.token == token,
            PasswordResetToken.is_used == False,
            PasswordResetToken.expires_at > datetime.now(timezone.utc),
        ).first()
        
        if not reset_token:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid or expired reset token",
            )
        
        # Get user
        user = db.query(User).filter(User.id == reset_token.user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User not found",
            )
        
        # Update password
        user.hashed_password = get_password_hash(new_password)
        reset_token.is_used = True
        reset_token.used_at = datetime.now(timezone.utc)
        
        db.commit()
        logger.info(f"Password reset successful for: {user.email}")
        
        return True
    
    @staticmethod
    def update_user_profile(
        db: Session,
        user_id: int,
        update_data: UserUpdate,
    ) -> User:
        """
        Update user profile
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # Update fields
        update_dict = update_data.dict(exclude_unset=True)
        for key, value in update_dict.items():
            setattr(user, key, value)
        
        user.updated_at = datetime.now(timezone.utc)
        db.commit()
        db.refresh(user)
        
        logger.info(f"User profile updated: {user.email}")
        return user
    
    @staticmethod
    def change_password(
        db: Session,
        user_id: int,
        change_data: ChangePassword,
    ) -> bool:
        """
        Change user password
        """
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )
        
        # Verify current password
        if not verify_password(change_data.current_password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect",
            )
        
        # Update password
        user.hashed_password = get_password_hash(change_data.new_password)
        user.updated_at = datetime.now(timezone.utc)
        db.commit()
        
        logger.info(f"Password changed for: {user.email}")
        return True
    
    @staticmethod
    def create_api_key(
        db: Session,
        user_id: int,
        api_key_data: ApiKeyCreate,
    ) -> Tuple[str, ApiKey]:
        """
        Create API key for user
        """
        # Generate API key
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)
        
        # Calculate expiry
        expires_at = None
        if api_key_data.expires_in_days:
            expires_at = datetime.now(timezone.utc) + timedelta(days=api_key_data.expires_in_days)
        
        # Save API key
        api_key_record = ApiKey(
            user_id=user_id,
            name=api_key_data.name,
            key_hash=key_hash,
            expires_at=expires_at,
        )
        
        db.add(api_key_record)
        db.commit()
        db.refresh(api_key_record)
        
        logger.info(f"API key created for user: {user_id}, name: {api_key_data.name}")
        return api_key, api_key_record
    
    @staticmethod
    def revoke_api_key(db: Session, user_id: int, api_key_id: int) -> bool:
        """
        Revoke API key
        """
        api_key = db.query(ApiKey).filter(
            ApiKey.id == api_key_id,
            ApiKey.user_id == user_id,
        ).first()
        
        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found",
            )
        
        api_key.is_active = False
        db.commit()
        
        logger.info(f"API key revoked: {api_key_id}")
        return True
    
    @staticmethod
    def get_user_api_keys(db: Session, user_id: int):
        """
        Get all API keys for user
        """
        return db.query(ApiKey).filter(
            ApiKey.user_id == user_id,
            ApiKey.is_active == True,
        ).order_by(ApiKey.created_at.desc()).all()


auth_service = AuthService()