# Keycloak SSO Integration

Connect WhisperLive to your company's Keycloak instance so employees can
authenticate with their existing corporate credentials.

## Overview

```
Employee              Keycloak              WhisperLive
   │                     │                      │
   ├──── Login ─────────►│                      │
   │◄─── JWT Token ──────┤                      │
   │                     │                      │
   ├──── Bearer Token ──────────────────────────►│
   │                     │                      ├── Validate JWT via JWKS
   │                     │                      ├── Extract role from claims
   │◄──── Transcription ────────────────────────┤
```

## Step 1: Create a Keycloak Client

In the Keycloak Admin Console:

1. Go to **Clients → Create client**
2. Fill in:
   - **Client ID**: `whisperlive`
   - **Client type**: `OpenID Connect`
   - **Root URL**: `https://your-whisperlive-host:9090`
3. Click **Next**, then:
   - **Client authentication**: Off (public client) — or On for confidential
   - **Authorization**: Off
   - **Standard flow**: ✅ (for web UI login)
   - **Direct access grants**: ✅ (for CLI/API usage)
4. Set **Valid redirect URIs**: `https://your-whisperlive-host:9090/*`
5. Click **Save**

## Step 2: Add a Role Mapper

WhisperLive reads roles from the `role` claim. Configure Keycloak to include it:

1. Go to **Clients → whisperlive → Client scopes**
2. Click the `whisperlive-dedicated` scope
3. **Add mapper → By configuration → User Realm Role**
4. Configure:
   - **Name**: `role-mapper`
   - **Token Claim Name**: `role`
   - **Claim JSON Type**: `String`
   - **Add to ID token**: ✅
   - **Add to access token**: ✅
   - **Multivalued**: Off (use the highest privilege role)
5. Click **Save**

### Create Realm Roles

In **Realm roles**, create these roles (matching WhisperLive's role system):

| Keycloak Role | WhisperLive Role | Permissions |
|---------------|-----------------|-------------|
| `admin` | admin | Full access + user management |
| `user` | user | Transcribe + read results |
| `readonly` | readonly | Read results only |

Assign roles to users under **Users → [user] → Role mappings**.

## Step 3: Start WhisperLive with Keycloak

```bash
# Your Keycloak realm URL
KEYCLOAK_URL="https://keycloak.yourcompany.com/realms/your-realm"

python run_server.py \
  --jwt_jwks_url "${KEYCLOAK_URL}/protocol/openid-connect/certs" \
  --jwt_issuer "${KEYCLOAK_URL}" \
  --jwt_audience "whisperlive" \
  --port 9090
```

### Docker Compose

```yaml
services:
  whisperlive:
    image: whisperlive:latest
    environment:
      - KEYCLOAK_URL=https://keycloak.yourcompany.com/realms/your-realm
    command: >
      python run_server.py
        --jwt_jwks_url "${KEYCLOAK_URL}/protocol/openid-connect/certs"
        --jwt_issuer "${KEYCLOAK_URL}"
        --jwt_audience "whisperlive"
        --port 9090
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

## Step 4: Get a Token and Use It

### Option A: Direct Access Grant (CLI/scripts)

```bash
# Get a token with username/password
TOKEN=$(curl -s -X POST \
  "https://keycloak.yourcompany.com/realms/your-realm/protocol/openid-connect/token" \
  -d "client_id=whisperlive" \
  -d "grant_type=password" \
  -d "username=alice" \
  -d "password=secret" \
  | jq -r '.access_token')

# Use it
curl -X POST http://localhost:9090/v1/audio/transcriptions \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@meeting.wav -F model=small
```

### Option B: Browser-based login (web UI)

If you're using the WhisperLive web UI, add a Keycloak JS adapter to handle
the login flow. Add this to the web UI's `index.html`:

```html
<script src="https://keycloak.yourcompany.com/js/keycloak.js"></script>
<script>
const keycloak = new Keycloak({
    url: 'https://keycloak.yourcompany.com',
    realm: 'your-realm',
    clientId: 'whisperlive'
});

keycloak.init({ onLoad: 'login-required' }).then(authenticated => {
    if (authenticated) {
        // Use keycloak.token as the Bearer token for API calls
        window.WHISPER_TOKEN = keycloak.token;

        // Auto-refresh token before expiry
        setInterval(() => {
            keycloak.updateToken(30).catch(() => keycloak.login());
        }, 60000);
    }
});
</script>
```

### Option C: Service account (machine-to-machine)

For automated pipelines, enable **Service accounts roles** on the client:

1. **Clients → whisperlive → Settings → Client authentication**: On
2. **Service accounts roles**: ✅
3. Assign the `user` role under **Service account roles**

```bash
TOKEN=$(curl -s -X POST \
  "https://keycloak.yourcompany.com/realms/your-realm/protocol/openid-connect/token" \
  -d "client_id=whisperlive" \
  -d "client_secret=YOUR_CLIENT_SECRET" \
  -d "grant_type=client_credentials" \
  | jq -r '.access_token')
```

## Combined Mode: Keycloak + Local API Keys

For a hybrid setup where most users log in via Keycloak but some service
accounts use local API keys:

```bash
python run_server.py \
  --user_store users.json \
  --jwt_jwks_url "https://keycloak.yourcompany.com/realms/your-realm/protocol/openid-connect/certs" \
  --jwt_issuer "https://keycloak.yourcompany.com/realms/your-realm" \
  --jwt_audience "whisperlive"
```

Auth order: local API key → Keycloak JWT → fallback `--api_key`.

## Verify It Works

```bash
# 1. Get a token
TOKEN=$(curl -s -X POST \
  "https://keycloak.yourcompany.com/realms/your-realm/protocol/openid-connect/token" \
  -d "client_id=whisperlive" \
  -d "grant_type=password" \
  -d "username=testuser" \
  -d "password=testpass" \
  | jq -r '.access_token')

# 2. Decode it (inspect claims)
echo "$TOKEN" | cut -d. -f2 | base64 -d 2>/dev/null | jq .

# 3. Hit the health endpoint (no auth needed)
curl http://localhost:9090/health

# 4. Transcribe (auth required)
curl -X POST http://localhost:9090/v1/audio/transcriptions \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@test.wav -F model=small
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `401 Invalid or missing API key` | Check token is valid: decode and verify `exp`, `iss`, `aud` claims |
| `JWT token expired` | Tokens are short-lived (default 5min). Refresh with `keycloak.updateToken()` or re-authenticate |
| `Invalid JWT token` | Verify `--jwt_issuer` matches the `iss` claim exactly (trailing slashes matter) |
| Role not recognized | Ensure the role mapper outputs `admin`, `user`, or `readonly` as the `role` claim |
| JWKS connection refused | Ensure WhisperLive can reach the Keycloak URL. In Docker, use the container network hostname |
| Self-signed Keycloak cert | Set `PYTHONHTTPSVERIFY=0` or mount the CA cert into the container |

## Security Notes

- Keycloak access tokens are **short-lived** (5 min default). Configure token
  lifespan in **Realm settings → Tokens**.
- Use **HTTPS** for both Keycloak and WhisperLive in production.
- For the web UI, use **PKCE** (Proof Key for Code Exchange) — Keycloak
  supports it out of the box with public clients.
- Rotate the client secret periodically if using a confidential client.
