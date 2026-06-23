from contextlib import asynccontextmanager

from analytics import router as analytics_router
from auth.dependencies import get_current_user
from auth.router import router as auth_router
from chat import router as chat_router
from config import settings
from db import init_collection
from embeddings import init_model
from fastapi import Depends, FastAPI
from gaps import router as gaps_router
from ingest import router as ingest_router
from portfolio import router as portfolio_router
from search import router as search_router
from timeline import router as timeline_router
from user_profile import router as user_profile_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_model(settings.embedding_model)
    init_collection(
        host=settings.chroma_host,
        port=settings.chroma_port,
        auth_token=settings.chroma_auth_token,
        collection_name=settings.collection_name,
    )
    yield


app = FastAPI(title="Personal Knowledge OS", version="0.1.0", lifespan=lifespan)

_auth = [Depends(get_current_user)]

app.include_router(ingest_router, dependencies=_auth)
app.include_router(search_router, dependencies=_auth)
app.include_router(chat_router, dependencies=_auth)
app.include_router(analytics_router, dependencies=_auth)
app.include_router(gaps_router, dependencies=_auth)
app.include_router(user_profile_router, dependencies=_auth)
app.include_router(portfolio_router, dependencies=_auth)
app.include_router(timeline_router, dependencies=_auth)
app.include_router(auth_router)  # 인증 없이 공개


@app.get("/health")
async def health():
    return {"status": "ok"}
