import datetime
import logging
import os
from zoneinfo import ZoneInfo
from fastapi import FastAPI, Depends, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

import httpx
import jwt
from jwt.algorithms import RSAAlgorithm
import requests

from mcp.server.fastmcp import FastMCP
from mcp.server.sse import SseServerTransport
from dotenv import load_dotenv

load_dotenv()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcp-server")

# -----------------------------
# Authentication setup
# -----------------------------

AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN")

API_AUDIENCE = os.getenv("AUTH_AUDIENCE") #<your-domain>/api/v2/

ALGORITHMS = ["RS256"]

JWKS_URL = f"https://{AUTH0_DOMAIN}/.well-known/jwks.json"

JWKS = requests.get(JWKS_URL,verify=False).json()

security = HTTPBearer()


def verify_jwt(token: str):
    """
    Verifies the validity of a JWT token using Auth0's JWKS public keys.
    Returns the decoded payload if valid, otherwise raises an HTTPException.
    """
    try:
        
        header = jwt.get_unverified_header(token)
        kid = header["kid"]  # Get the unique Key ID
        
        key = next((k for k in JWKS["keys"] if k["kid"] == kid), None)
        if not key:            
            raise HTTPException(status_code=401, detail="Invalid auth key")
        
        public_key = RSAAlgorithm.from_jwk(key)

        
        payload = jwt.decode(
            token,
            public_key,
            algorithms=ALGORITHMS,   # e.g. ["RS256"]
            audience=API_AUDIENCE,   # expected audience, usually your API identifier
            issuer=f"https://{AUTH0_DOMAIN}/"  # expected issuer, e.g. "https://your-tenant.auth0.com/"
        )
        
        return payload

    except Exception as e:        
        logger.error(f"JWT validation error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
        
        
# ---------- MCP and tools ----------
mcp = FastMCP(name="Weather and Time SSE Server")

transport = SseServerTransport("/messages/")


@mcp.tool()
def TimeTool(input_timezone: str = None):
    current_time = datetime.datetime.now()
    if input_timezone:
        current_time = current_time.astimezone(ZoneInfo(input_timezone))
    return f"The current time is {current_time}."


@mcp.tool()
def weather_tool(location: str):
    api_key = os.getenv("OPENWEATHERMAP_API_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()
    if data.get("cod") == 200:
        temp = data["main"]["temp"]
        description = data["weather"][0]["description"]
        return f"The weather in {location} is {description} with {temp}°C."
    return f"Couldn't fetch weather for {location}."



# ---------- ASGI wrapper to protect transport.handle_post_message ----------
def auth_asgi_wrapper(asgi_app):
    """
    ASGI middleware-like wrapper that enforces JWT Bearer token authentication 
    for incoming HTTP requests before passing them to the underlying ASGI app.

    - It only applies authentication checks for HTTP traffic (e.g. REST endpoints).
    - It specifically ensures that requests carry a valid 'Authorization: Bearer <token>' header.
    - If the JWT is missing, invalid, or expired, it responds immediately with HTTP 401.
    - If the JWT is valid, the request is forwarded to the wrapped ASGI app.
    """

    print(asgi_app)
    async def app(scope, receive, send):
        """
        ASGI entry point function. Every incoming request (HTTP, WebSocket, lifespan, etc.)
        passes through here before reaching the main application.
        """

        # Check if the request is HTTP. Other protocols (WebSocket, lifespan) skip authentication.
        if scope["type"] == "http":
            
            headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}            
            auth_hdr = headers.get("authorization")
            
            if not auth_hdr or not auth_hdr.lower().startswith("bearer "):                
                resp = JSONResponse({"detail": "Not authenticated"}, status_code=401)
                await resp(scope, receive, send)
                return
            
            token = auth_hdr.split(" ", 1)[1]

            try:            
                verify_jwt(token)
            except HTTPException as he:                
                resp = JSONResponse({"detail": he.detail}, status_code=he.status_code)
                await resp(scope, receive, send)
                return

            except Exception:                
                resp = JSONResponse({"detail": "Invalid or expired token"}, status_code=401)
                await resp(scope, receive, send)
                return
        
        await asgi_app(scope, receive, send)

    return app
    
# ---------- FastAPI app (use FastAPI route so Depends() runs) ----------
app = FastAPI()

@app.get("/health")
def health():
    return {"status": "OK"}


@app.get("/sse")
async def handle_sse(request: Request):
    try:
        auth_hdr = request.headers.get("authorization")
        token = None
        if auth_hdr and auth_hdr.lower().startswith("bearer "):
            token = auth_hdr.split(" ", 1)[1]
        else:
            token = request.query_params.get("access_token")

        if not token:
            raise HTTPException(status_code=401, detail="Missing token")
        
        verify_jwt(token)
        
        async with transport.connect_sse(request.scope, request.receive, request._send) as (in_stream, out_stream):
            await mcp._mcp_server.run(in_stream, out_stream, mcp._mcp_server.create_initialization_options())
    except Exception as httpx_e:
        logger.error(f"Error while validating token. {httpx_e}")
        return Response(status_code=401,content="Invalid or expired token")
        

# mount the POST handler (ASGI app) but wrap it so it requires auth
app.mount("/messages/", auth_asgi_wrapper(transport.handle_post_message))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8100)