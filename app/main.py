import retrieval_core
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.deps import get_app_settings
from app.settings import Settings

app = FastAPI(title="ShopRank")

# We apply CORS middleware only on startup based on settings, but we can't do it dynamically inside a route.
# We'll need settings at import time, or we can fetch them via get_settings() directly.
settings = get_app_settings()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping(settings: Settings = Depends(get_app_settings)) -> dict[str, str]:  # noqa: B008
    return {
        "status": "pong",
        "version": settings.git_sha,
        "retrieval_core": retrieval_core.__version__
    }

