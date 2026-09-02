from contextlib import asynccontextmanager

import retrieval_core
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.deps import get_app_settings
from app.limiter import limiter
from app.routes import ablation, examples, search
from app.settings import Settings
from core.pipeline import close_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_pipeline()


settings = get_app_settings()

docs_url = "/docs" if settings.environment == "local" else None
redoc_url = "/redoc" if settings.environment == "local" else None
openapi_url = "/openapi.json" if settings.environment == "local" else None

app = FastAPI(
    title="ShopRank",
    lifespan=lifespan,
    docs_url=docs_url,
    redoc_url=redoc_url,
    openapi_url=openapi_url
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

app.include_router(search.router, prefix="/api")
app.include_router(examples.router, prefix="/api")
app.include_router(ablation.router, prefix="/api")


from typing import Annotated


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "service": "ShopRank API",
        "docs": "/docs",
        "health": "/healthz",
        "ui": "http://localhost:4200",
    }


@app.get("/healthz")
async def healthz(
    settings: Annotated[Settings, Depends(get_app_settings)] = None,  # type: ignore[assignment]
) -> dict[str, str]:
    """Liveness probe. Strictly ZERO database calls."""
    return {
        "status": "ok",
        "version": settings.git_sha,
    }


@app.get("/ping")
async def ping(
    settings: Annotated[Settings, Depends(get_app_settings)] = None,  # type: ignore[assignment]
) -> dict[str, str]:
    return {
        "status": "pong",
        "version": settings.git_sha,
        "retrieval_core": retrieval_core.__version__,
    }
