"""
FastAPI OAuth2 + JWT with MongoDB (Motor)
Router version: register -> login -> protected route
"""

import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr, Field
from passlib.context import CryptContext
from jose import JWTError, jwt
import motor.motor_asyncio
from bson import ObjectId

# -----------------------
# Router
# -----------------------
router = APIRouter()

# -----------------------
# Configuration
# -----------------------
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "fastapi_auth_db")
USERS_COLL = "users"

SECRET_KEY = os.getenv("SECRET_KEY", "change_this_secret_in_production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# -----------------------
# MongoDB client
# -----------------------
client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
db = client[DB_NAME]
users_collection = db[USERS_COLL]

# -----------------------
# Password hashing
# -----------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    return pwd_context.hash(password[:72])

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# -----------------------
# Pydantic models
# -----------------------
class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)

class UserInDB(BaseModel):
    id: str
    username: str
    email: EmailStr
    hashed_password: str
    is_active: bool = True

class UserPublic(BaseModel):
    id: str
    username: str
    email: EmailStr
    is_active: bool = True

class Token(BaseModel):
    access_token: str
    token_type: str

# -----------------------
# Helper functions
# -----------------------
def objid_to_str(o: ObjectId) -> str:
    return str(o)

async def get_user_by_username(username: str) -> Optional[UserInDB]:
    doc = await users_collection.find_one({"username": username})
    if not doc:
        return None
    return UserInDB(
        id=objid_to_str(doc["_id"]),
        username=doc["username"],
        email=doc["email"],
        hashed_password=doc["hashed_password"],
        is_active=doc.get("is_active", True)
    )

async def get_user_by_id(user_id: str) -> Optional[UserInDB]:
    try:
        _id = ObjectId(user_id)
    except Exception:
        return None
    doc = await users_collection.find_one({"_id": _id})
    if not doc:
        return None
    return UserInDB(
        id=objid_to_str(doc["_id"]),
        username=doc["username"],
        email=doc["email"],
        hashed_password=doc["hashed_password"],
        is_active=doc.get("is_active", True)
    )

async def create_user(user: UserCreate) -> UserPublic:
    existing = await users_collection.find_one({"$or": [{"username": user.username}, {"email": user.email}]})
    if existing:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    hashed = hash_password(user.password)
    res = await users_collection.insert_one({
        "username": user.username,
        "email": user.email,
        "hashed_password": hashed,
        "is_active": True,
        "created_at": datetime.utcnow()
    })
    return UserPublic(id=objid_to_str(res.inserted_id), username=user.username, email=user.email, is_active=True)

# -----------------------
# JWT Auth
# -----------------------
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    now = datetime.utcnow()
    expire = now + (expires_delta if expires_delta else timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "iat": now})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")  # token URL is now /auth/token

async def authenticate_user(username: str, password: str) -> Optional[UserInDB]:
    user = await get_user_by_username(username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = payload.get("sub")
        if not sub:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = await get_user_by_id(sub)
    if not user or not user.is_active:
        raise credentials_exception
    return user

# -----------------------
# Routes
# -----------------------
@router.post("/register", response_model=UserPublic, status_code=201)
async def register(user: UserCreate):
    """Register a new user"""
    return await create_user(user)

@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login to get JWT token"""
    user = await authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=401, detail="Incorrect username or password", headers={"WWW-Authenticate": "Bearer"})
    token = create_access_token(data={"sub": user.id})
    return {"access_token": token, "token_type": "bearer"}

@router.get("/users/me", response_model=UserPublic)
async def read_users_me(current_user: UserInDB = Depends(get_current_user)):
    """Protected route"""
    return UserPublic(id=current_user.id, username=current_user.username, email=current_user.email, is_active=current_user.is_active)
