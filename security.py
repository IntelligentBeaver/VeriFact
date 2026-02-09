# security.py
from passlib.context import CryptContext

# use bcrypt_sha256 to avoid bcrypt's 72-byte limit
pwd_context = CryptContext(schemes=["bcrypt_sha256"], deprecated="auto")

def hash_password(password: str) -> str:
    # passlib will handle string -> bytes and the sha256 pre-hash
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
