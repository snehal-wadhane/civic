# auth.py - Authentication and Authorization
from datetime import datetime, timedelta
from typing import Optional
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel
from config import settings
from database import supabase
import logging

logger = logging.getLogger(__name__)

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Models
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None
    uid: Optional[int] = None
    role: Optional[str] = None

class User(BaseModel):
    uid: int
    email: str
    full_name: Optional[str] = None
    role: str = "citizen"
    department_id: Optional[int] = None

# Password utilities
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

# Token utilities
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> TokenData:
    """Verify and decode JWT token"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        uid: int = payload.get("uid")
        role: str = payload.get("role")
        
        if email is None:
            raise credentials_exception
            
        token_data = TokenData(email=email, uid=uid, role=role)
        return token_data
    except JWTError:
        raise credentials_exception

# User authentication
async def authenticate_user(email: str, password: str) -> Optional[User]:
    """Authenticate user with email and password"""
    try:
        # Fetch user from database
        response = supabase.table("users").select("*").eq("email", email).execute()
        
        if not response.data:
            return None
        
        user_data = response.data[0]
        
        # Verify password
        if not verify_password(password, user_data.get("password_hash")):
            return None
        
        return User(
            uid=user_data.get("uid"),
            email=user_data.get("email"),
            full_name=user_data.get("full_name"),
            role=user_data.get("role", "citizen"),
            department_id=user_data.get("department_id")
        )
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        return None

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get current authenticated user from token"""
    token_data = verify_token(token)
    
    # Fetch user from database
    response = supabase.table("users").select("*").eq("uid", token_data.uid).execute()
    
    if not response.data:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    user_data = response.data[0]
    return User(
        uid=user_data.get("uid"),
        email=user_data.get("email"),
        full_name=user_data.get("full_name"),
        role=user_data.get("role", "citizen"),
        department_id=user_data.get("department_id")
    )

async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    return current_user

# Role-based authorization
def require_role(allowed_roles: list):
    """Decorator to require specific roles"""
    async def role_checker(current_user: User = Depends(get_current_user)):
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker

# Specific role dependencies
async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Require admin role"""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user

async def get_department_staff(current_user: User = Depends(get_current_user)) -> User:
    """Require department staff or admin role"""
    if current_user.role not in ["admin", "department_staff"]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Department staff access required"
        )
    return current_user

# Optional authentication (for public + authenticated endpoints)
async def get_current_user_optional(token: str = Depends(oauth2_scheme)) -> Optional[User]:
    """Get current user if authenticated, None otherwise"""
    if not token:
        return None
    
    try:
        return await get_current_user(token)
    except HTTPException:
        return None
