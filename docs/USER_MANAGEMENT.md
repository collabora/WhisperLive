# User Management & Access Control

WhisperLive includes a built-in multi-user access control system with role-based
permissions, per-user API keys, rate limiting, usage quotas, and JWT support for
enterprise identity providers.

## Quick Start

### 1. Enable user management

```bash
python run_server.py --user_store users.json
```

This creates a `users.json` file to store user accounts. On first run the file
is empty — you'll bootstrap your first admin user with the helper script below.

### 2. Create an admin user

```bash
python -c "
from whisper_live.acl import UserStore, Role
store = UserStore('users.json')
user, key = store.create_user('Admin', Role.ADMIN)
print(f'User ID:  {user.user_id}')
print(f'API Key:  {key}')
print('Save this key — it is only shown once.')
"
```

### 3. Use the API key

```bash
# Transcribe with your personal key
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Authorization: Bearer wl_YOUR_KEY_HERE" \
  -F file=@audio.wav -F model=small
```

### 4. Manage users via the Admin API

```bash
ADMIN_KEY="wl_YOUR_ADMIN_KEY"

# List users
curl -H "Authorization: Bearer $ADMIN_KEY" http://localhost:8000/v1/admin/users

# Create a user
curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
  http://localhost:8000/v1/admin/users \
  -F name="Alice" -F role=user -F rate_limit_rpm=120 -F quota_minutes=500

# Update a user
curl -X PATCH -H "Authorization: Bearer $ADMIN_KEY" \
  http://localhost:8000/v1/admin/users/USER_ID \
  -F rate_limit_rpm=200

# Rotate a user's API key
curl -X POST -H "Authorization: Bearer $ADMIN_KEY" \
  http://localhost:8000/v1/admin/users/USER_ID/rotate-key

# Delete a user
curl -X DELETE -H "Authorization: Bearer $ADMIN_KEY" \
  http://localhost:8000/v1/admin/users/USER_ID
```

---

## Roles

| Role | Transcribe | Read results | Admin API |
|------|-----------|-------------|-----------|
| `admin` | ✅ | ✅ | ✅ |
| `user` | ✅ | ✅ | ❌ |
| `readonly` | ❌ | ✅ | ❌ |

---

## Rate Limits & Quotas

Each user has individual limits:

- **`rate_limit_rpm`** — Maximum requests per minute (default: 60). The server
  returns `429 Too Many Requests` when exceeded.
- **`quota_minutes`** — Monthly audio minutes quota (default: 0 = unlimited).
  Returns `429` with `"Monthly quota exceeded"` when used up.

Reset monthly usage for all users:

```bash
python -c "
from whisper_live.acl import UserStore
store = UserStore('users.json')
store.reset_monthly_usage()
print('Done')
"
```

---

## API Key Security

- Keys are prefixed with `wl_` for easy identification in logs and secrets
  scanners.
- Keys are **SHA-256 hashed** before storage — the plaintext key is only shown
  once at creation time.
- Use **`rotate-key`** to invalidate a compromised key and generate a new one.
- Disabled users (`enabled: false`) cannot authenticate even with a valid key.

---

## JWT Authentication

For enterprise deployments with an existing identity provider (AWS Cognito,
Auth0, Keycloak, Okta, etc.), WhisperLive can validate JWT tokens directly.

### RS256 (Cognito, Auth0, Keycloak)

```bash
python run_server.py \
  --jwt_jwks_url "https://cognito-idp.us-east-1.amazonaws.com/POOL_ID/.well-known/jwks.json" \
  --jwt_audience "your-client-id" \
  --jwt_issuer "https://cognito-idp.us-east-1.amazonaws.com/POOL_ID"
```

### HS256 (shared secret)

```bash
python run_server.py --jwt_secret "your-shared-secret"
```

### Combined mode

You can use both a local user store **and** JWT simultaneously. The auth
middleware tries them in order:

1. **User store** — check if the Bearer token matches a local API key
2. **JWT** — validate as a JWT token
3. **Fallback API key** — check against `--api_key` (backwards compatible)

```bash
python run_server.py \
  --user_store users.json \
  --jwt_jwks_url "https://..." \
  --api_key "legacy-shared-key"
```

### JWT claims mapping

The system extracts user info from standard JWT claims:

| Claim | Used for |
|-------|----------|
| `sub` | User ID |
| `email` | Email address |
| `name` or `cognito:username` | Display name |
| `custom:role` or `role` | Role (admin/user/readonly) |
| `cognito:groups` or `groups` | Group memberships |

---

## CLI Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--user_store` | str | None | Path to JSON user store file |
| `--jwt_jwks_url` | str | None | JWKS URL for RS256 JWT validation |
| `--jwt_secret` | str | None | Shared secret for HS256 JWT validation |
| `--jwt_audience` | str | None | Expected JWT audience claim |
| `--jwt_issuer` | str | None | Expected JWT issuer claim |
| `--api_key` | str | None | Simple shared API key (legacy, still supported) |

---

## Admin API Reference

All admin endpoints require a Bearer token from a user with the `admin` role.

### `GET /v1/admin/users`

List all users. API key hashes are redacted from the response.

**Response:**
```json
{
  "users": [
    {
      "user_id": "a1b2c3d4",
      "name": "Alice",
      "role": "user",
      "rate_limit_rpm": 60,
      "quota_minutes": 500,
      "used_minutes": 12.5,
      "enabled": true,
      "created_at": 1700000000.0,
      "last_used": 1700001000.0
    }
  ]
}
```

### `POST /v1/admin/users`

Create a new user. Returns the API key — **save it, it's only shown once**.

**Form fields:** `name` (required), `role`, `rate_limit_rpm`, `quota_minutes`

**Response:**
```json
{
  "user_id": "a1b2c3d4",
  "name": "Alice",
  "role": "user",
  "api_key": "wl_abc123...",
  "rate_limit_rpm": 60,
  "quota_minutes": 500
}
```

### `PATCH /v1/admin/users/{user_id}`

Update user settings. Only send the fields you want to change.

**Form fields:** `name`, `role`, `rate_limit_rpm`, `quota_minutes`, `enabled`

### `DELETE /v1/admin/users/{user_id}`

Delete a user and revoke their API key immediately.

### `POST /v1/admin/users/{user_id}/rotate-key`

Generate a new API key for a user. The old key is invalidated immediately.

**Response:**
```json
{
  "user_id": "a1b2c3d4",
  "api_key": "wl_newkey456..."
}
```

---

## Docker / Docker Compose

Mount the user store file as a volume so it persists across container restarts:

```yaml
services:
  whisperlive:
    image: whisperlive:latest
    volumes:
      - ./users.json:/app/users.json
    command: >
      python run_server.py
        --user_store /app/users.json
        --port 9090
```

For JWT with Cognito in the Terraform/ECS deployment, pass the JWKS URL as an
environment variable or through AWS Secrets Manager.

---

## Architecture

```
Client Request
    │
    ▼
┌─────────────────────────┐
│   Auth Middleware        │
│                         │
│  1. UserStore.auth()    │──→ Per-user rate limit check
│  2. JWTValidator.val()  │──→ Claims extracted
│  3. Fallback API key    │──→ Legacy compat
│                         │
│  403 if admin-only      │
│  429 if rate/quota hit  │
│  401 if no valid cred   │
└─────────────────────────┘
    │
    ▼
  Endpoint handler
```

---

## Requirements

The ACL module uses only the Python standard library. JWT validation requires
an additional package:

```bash
pip install PyJWT[crypto]
```

This is included in `requirements/server.txt`.
