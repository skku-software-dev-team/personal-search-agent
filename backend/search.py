from auth.dependencies import get_current_user
from database.models import User
from db import get_collection
from embeddings import get_model
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

router = APIRouter()


# ── Response models ───────────────────────────────────────────────────────────


class SearchResult(BaseModel):
    text: str
    source: str  # "local" | "gdrive" | "notion"
    file_name: str
    file_path: str
    created_at: str
    chunk_index: int
    score: float  # 코사인 유사도 (0~1, 높을수록 유사)


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]
    total: int


# ── Core search logic (search.py 라우트와 chat.py가 함께 재사용) ──────────────────


def search_documents(
    query: str, user_id: str, top_k: int = 10, source: str | None = None
) -> list[SearchResult]:
    query_vector = get_model().encode([query], show_progress_bar=False).tolist()

    where = {"source": source} if source else None

    raw = get_collection(user_id).query(
        query_embeddings=query_vector,
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    return [
        SearchResult(
            text=doc,
            source=meta.get("source", ""),
            file_name=meta.get("file_name", ""),
            file_path=meta.get("file_path", ""),
            created_at=meta.get("created_at", ""),
            chunk_index=meta.get("chunk_index", 0),
            score=round(1 - dist, 4),  # ChromaDB cosine distance → similarity
        )
        for doc, meta, dist in zip(
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        )
    ]


# ── Routes ────────────────────────────────────────────────────────────────────


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., description="검색 쿼리"),
    top_k: int = Query(10, ge=1, le=50, description="반환할 최대 결과 수"),
    source: str | None = Query(None, description="소스 필터 (local | gdrive | notion)"),
    current_user: User = Depends(get_current_user),
):
    if not q.strip():
        raise HTTPException(status_code=422, detail="검색어를 입력해주세요.")

    results = search_documents(q, user_id=current_user.id, top_k=top_k, source=source)
    return SearchResponse(query=q, results=results, total=len(results))
