from contextlib import asynccontextmanager

import retrieval_core
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.deps import get_app_settings
from app.routes import ablation, examples, search
from app.settings import Settings
from core.pipeline import close_pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    await close_pipeline()


app = FastAPI(title="ShopRank", lifespan=lifespan)

settings = get_app_settings()

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


@app.get("/ping")
async def ping(settings: Settings = Depends(get_app_settings)) -> dict[str, str]:  # noqa: B008
    return {
        "status": "pong",
        "version": settings.git_sha,
        "retrieval_core": retrieval_core.__version__,
    }
