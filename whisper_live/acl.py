"""
User management and access control for WhisperLive.

Supports:
- Local JSON file user store (for dev/staging)
- JWT token validation (for Cognito, Auth0, Keycloak, etc.)
- Role-based access: admin, user, readonly
- Per-user API keys with rate limits and usage quotas
"""

import hashlib
import hmac
import json
import logging
import os
import secrets
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

    def can_transcribe(self) -> bool:
        return self in (Role.ADMIN, Role.USER)

    def can_admin(self) -> bool:
        return self == Role.ADMIN

    def can_read(self) -> bool:
        return True


@dataclass
class User:
    user_id: str
    name: str
    role: Role
    api_key_hash: str  # SHA-256 hash of the API key
    rate_limit_rpm: int = 60  # requests per minute
    quota_minutes: int = 0  # monthly audio minutes quota (0=unlimited)
    used_minutes: float = 0.0  # audio minutes used this month
    enabled: bool = True
    created_at: float = field(default_factory=time.time)
    last_used: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "User":
        d["role"] = Role(d["role"])
        return cls(**d)


def _hash_key(api_key: str) -> str:
    """Hash an API key with SHA-256."""
    return hashlib.sha256(api_key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"wl_{secrets.token_urlsafe(32)}"


class UserStore:
    """Manage users in a local JSON file."""

    def __init__(self, path: str = "users.json"):
        self._path = path
        self._lock = threading.Lock()
        self._users: Dict[str, User] = {}
        self._key_index: Dict[str, str] = {}  # key_hash -> user_id
        self._rate_buckets: Dict[str, list] = {}  # user_id -> list of timestamps
        self._load()

    def _load(self):
        if os.path.exists(self._path):
            try:
                with open(self._path) as f:
                    data = json.load(f)
                for ud in data.get("users", []):
                    user = User.from_dict(ud)
                    self._users[user.user_id] = user
                    self._key_index[user.api_key_hash] = user.user_id
                logger.info(f"Loaded {len(self._users)} users from {self._path}")
            except Exception as e:
                logger.error(f"Failed to load user store: {e}")

    def _save(self):
        try:
            data = {"users": [u.to_dict() for u in self._users.values()]}
            with open(self._path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save user store: {e}")

    def create_user(self, name: str, role: Role = Role.USER,
                    rate_limit_rpm: int = 60,
                    quota_minutes: int = 0) -> tuple:
        """Create a new user. Returns (user, api_key)."""
        user_id = secrets.token_hex(8)
        api_key = generate_api_key()
        key_hash = _hash_key(api_key)

        user = User(
            user_id=user_id,
            name=name,
            role=role,
            api_key_hash=key_hash,
            rate_limit_rpm=rate_limit_rpm,
            quota_minutes=quota_minutes,
        )

        with self._lock:
            self._users[user_id] = user
            self._key_index[key_hash] = user_id
            self._save()

        logger.info(f"Created user '{name}' ({user_id}) with role {role.value}")
        return user, api_key

    def authenticate(self, api_key: str) -> Optional[User]:
        """Authenticate by API key. Returns User or None."""
        key_hash = _hash_key(api_key)
        with self._lock:
            user_id = self._key_index.get(key_hash)
            if not user_id:
                return None
            user = self._users.get(user_id)
            if user and user.enabled:
                user.last_used = time.time()
                return user
            return None

    def get_user(self, user_id: str) -> Optional[User]:
        with self._lock:
            return self._users.get(user_id)

    def list_users(self) -> List[dict]:
        with self._lock:
            return [u.to_dict() for u in self._users.values()]

    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return None
            for key, value in kwargs.items():
                if key == "role":
                    value = Role(value)
                if hasattr(user, key) and key not in ("user_id", "api_key_hash", "created_at"):
                    setattr(user, key, value)
            self._save()
            return user

    def delete_user(self, user_id: str) -> bool:
        with self._lock:
            user = self._users.pop(user_id, None)
            if user:
                self._key_index.pop(user.api_key_hash, None)
                self._save()
                return True
            return False

    def rotate_key(self, user_id: str) -> Optional[str]:
        """Generate a new API key for a user. Returns the new key."""
        with self._lock:
            user = self._users.get(user_id)
            if not user:
                return None
            # Remove old key from index
            self._key_index.pop(user.api_key_hash, None)
            # Generate new key
            new_key = generate_api_key()
            user.api_key_hash = _hash_key(new_key)
            self._key_index[user.api_key_hash] = user_id
            self._save()
            return new_key

    def check_rate_limit(self, user_id: str) -> bool:
        """Check if user is within rate limit. Returns True if allowed."""
        with self._lock:
            user = self._users.get(user_id)
            if not user or user.rate_limit_rpm <= 0:
                return True

            now = time.time()
            bucket = self._rate_buckets.setdefault(user_id, [])
            # Remove entries older than 60s
            bucket[:] = [t for t in bucket if t > now - 60]

            if len(bucket) >= user.rate_limit_rpm:
                return False
            bucket.append(now)
            return True

    def check_quota(self, user_id: str) -> bool:
        """Check if user is within monthly quota. Returns True if allowed."""
        with self._lock:
            user = self._users.get(user_id)
            if not user or user.quota_minutes <= 0:
                return True  # Unlimited
            return user.used_minutes < user.quota_minutes

    def track_usage(self, user_id: str, audio_minutes: float):
        """Track audio minutes used by a user."""
        with self._lock:
            user = self._users.get(user_id)
            if user:
                user.used_minutes += audio_minutes
                self._save()

    def reset_monthly_usage(self):
        """Reset all users' monthly usage counters."""
        with self._lock:
            for user in self._users.values():
                user.used_minutes = 0.0
            self._save()
        logger.info("Reset monthly usage for all users")


class JWTValidator:
    """Validate JWT tokens from external identity providers.

    Supports RS256 (Cognito, Auth0, Keycloak) and HS256 tokens.
    For RS256, provide the JWKS URL; for HS256, provide the secret.
    """

    def __init__(self, jwks_url: Optional[str] = None,
                 secret: Optional[str] = None,
                 audience: Optional[str] = None,
                 issuer: Optional[str] = None):
        self._jwks_url = jwks_url
        self._secret = secret
        self._audience = audience
        self._issuer = issuer
        self._jwks_cache = None
        self._jwks_cache_time = 0

    def validate(self, token: str) -> Optional[dict]:
        """Validate a JWT token. Returns claims dict or None.

        Requires PyJWT: pip install PyJWT[crypto]
        """
        try:
            import jwt as pyjwt
        except ImportError:
            logger.error("PyJWT required for JWT validation: pip install PyJWT[crypto]")
            return None

        try:
            if self._secret:
                # HS256 (simple shared secret)
                claims = pyjwt.decode(
                    token,
                    self._secret,
                    algorithms=["HS256"],
                    audience=self._audience,
                    issuer=self._issuer,
                )
                return claims

            if self._jwks_url:
                # RS256 (Cognito, Auth0, Keycloak)
                jwks_client = pyjwt.PyJWKClient(self._jwks_url)
                signing_key = jwks_client.get_signing_key_from_jwt(token)
                claims = pyjwt.decode(
                    token,
                    signing_key.key,
                    algorithms=["RS256"],
                    audience=self._audience,
                    issuer=self._issuer,
                )
                return claims

        except pyjwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
        except pyjwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
        except Exception as e:
            logger.error(f"JWT validation error: {e}")

        return None

    def get_user_info(self, claims: dict) -> dict:
        """Extract user info from JWT claims (works with Cognito, Auth0, Keycloak)."""
        return {
            "user_id": claims.get("sub", ""),
            "email": claims.get("email", ""),
            "name": claims.get("name", claims.get("cognito:username", "")),
            "role": claims.get("custom:role", claims.get("role", "user")),
            "groups": claims.get("cognito:groups", claims.get("groups", [])),
        }
