import asyncio
import json
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from openai import OpenAI

from config import settings
from db import get_collection
from gaps_agents import AgentContext, GapOrchestrator
from user_profile import UserProfile, load_profile

router = APIRouter()

RELATED_SIMILARITY_THRESHOLD = 0.6
REVIEW_THRESHOLD_DAYS = 180  # 6개월 이상 된 클러스터 → 복습 추천


# ── Request model ─────────────────────────────────────────────────────────────

class GapsRequest(BaseModel):
    goal: str | None = None
    fields: list[str] = []
    level: str | None = None
    timeline: str | None = None


# ── Response models ───────────────────────────────────────────────────────────

class ClusterSummary(BaseModel):
    cluster_id: int
    topic: str
    keywords: list[str]
    doc_count: int
    is_gap: bool
    severity: str             # "none" | "low" | "medium" | "critical"
    related_clusters: list[int]
    avg_age_days: int         # 클러스터 평균 문서 나이 (일)


class GapRecommendation(BaseModel):
    area: str
    severity: str             # "low" | "medium" | "critical"
    gap_type: str             # "missing" | "sparse" | "review"
    goal_relevance: str       # "high" | "medium" | "low"
    related_strong_topic: str | None
    reason: str
    recommendation: str       # 구체적인 실습 목표 / 마일스톤
    resources: list[str]      # 추천 학습 자료 (책/강의/공식문서)


class RoadmapPhase(BaseModel):
    phase: int
    period: str
    focus: str
    topics: list[str]


class GapsResponse(BaseModel):
    goal: str | None
    total_chunks: int
    n_clusters: int
    required_areas: list[str] | None
    clusters: list[ClusterSummary]
    gaps: list[GapRecommendation]
    roadmap: list[RoadmapPhase] | None
    agent_trace: list[dict] | None = None  # 각 Agent의 tool call 기록


# ── Helpers ───────────────────────────────────────────────────────────────────

def _gap_severity(doc_count: int, avg: float, has_related_strong: bool) -> str:
    if doc_count < avg * 0.3 and has_related_strong:
        return "critical"
    if doc_count < avg * 0.5:
        return "medium"
    if doc_count < avg * 0.7:
        return "low"
    return "none"


def _related_cluster_map(centers: np.ndarray) -> dict[int, list[int]]:
    n = len(centers)
    sim = cosine_similarity(centers)
    related: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] > RELATED_SIMILARITY_THRESHOLD:
                related[i].append(j)
                related[j].append(i)
    return related


def _cluster_avg_age_days(metadatas: list[dict]) -> int:
    """클러스터 내 문서들의 평균 나이(일) 계산. created_at 없으면 0."""
    now = datetime.now(timezone.utc)
    ages = []
    for meta in metadatas:
        created_at = (meta or {}).get("created_at")
        if not created_at:
            continue
        try:
            dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ages.append((now - dt).days)
        except (ValueError, AttributeError):
            pass
    return round(sum(ages) / len(ages)) if ages else 0


def _merge_profile(request: GapsRequest) -> UserProfile:
    saved = load_profile()
    return UserProfile(
        goal=request.goal if request.goal is not None else saved.goal,
        fields=request.fields if request.fields else saved.fields,
        level=request.level if request.level is not None else saved.level,
        timeline=request.timeline if request.timeline is not None else saved.timeline,
    )


def _goal_context(profile: UserProfile) -> str | None:
    if not profile.goal:
        return None
    lines = [f"- 커리어 목표: {profile.goal}"]
    if profile.fields:
        lines.append(f"- 관심 분야: {', '.join(profile.fields)}")
    if profile.level:
        lines.append(f"- 현재 레벨: {profile.level}")
    if profile.timeline:
        lines.append(f"- 목표 기간: {profile.timeline}")
    return "\n".join(lines)


def _llm_call(client: OpenAI, model: str, messages: list, temperature: float = 0.3) -> dict:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return json.loads(response.choices[0].message.content)
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="API 키가 올바르지 않습니다. .env 파일을 확인해주세요.")
    except openai.RateLimitError as e:
        msg = str(e)
        import re
        m = re.search(r'try again in (\d+m[\d.]+s)', msg)
        wait = f" ({m.group(1)} 후 재시도)" if m else ""
        raise HTTPException(status_code=429, detail=f"Groq 일일 토큰 한도 초과{wait}. 내일 다시 시도해주세요.")
    except openai.APIConnectionError:
        raise HTTPException(status_code=502, detail="LLM 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.")
    except openai.OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"LLM API 오류: {e}")


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/gaps", response_model=GapsResponse)
async def post_gaps(body: GapsRequest = GapsRequest()):
    profile = _merge_profile(body)
    goal_ctx = _goal_context(profile)

    collection = get_collection()
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    docs = result.get("documents") or []
    embeddings = result.get("embeddings") or []
    metadatas = result.get("metadatas") or []

    if not docs or len(docs) < 3:
        raise HTTPException(
            status_code=422,
            detail="분석에 필요한 문서가 부족합니다. 최소 3개 이상의 청크가 필요합니다.",
        )

    total = len(docs)
    n_clusters = min(8, max(3, total // 10))

    X = np.array(embeddings, dtype=float)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = kmeans.fit_predict(X)

    clusters_docs: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    clusters_metas: dict[int, list[dict]] = {i: [] for i in range(n_clusters)}
    for doc, meta, label in zip(docs, metadatas, labels):
        cid = int(label)
        clusters_docs[cid].append(doc)
        clusters_metas[cid].append(meta or {})

    cluster_sizes = {cid: len(cdocs) for cid, cdocs in clusters_docs.items()}
    avg_size = total / n_clusters

    related_map = _related_cluster_map(kmeans.cluster_centers_)
    strong_cluster_ids = {cid for cid, sz in cluster_sizes.items() if sz >= avg_size}

    def has_related_strong(cid: int) -> bool:
        return any(r in strong_cluster_ids for r in related_map[cid])

    severities = {
        cid: _gap_severity(cluster_sizes[cid], avg_size, has_related_strong(cid))
        for cid in range(n_clusters)
    }
    gap_cluster_ids = {cid for cid, sev in severities.items() if sev != "none"}

    # ── 시간 가중치: 오래된 클러스터 탐지 ────────────────────────────────────────
    cluster_avg_age = {
        cid: _cluster_avg_age_days(clusters_metas[cid])
        for cid in range(n_clusters)
    }
    # sparse gap이 아닌데 오래된 클러스터 → 복습 추천 대상
    review_cluster_ids = {
        cid for cid in range(n_clusters)
        if cid not in gap_cluster_ids
        and cluster_avg_age[cid] >= REVIEW_THRESHOLD_DAYS
    }

    def cluster_sample(cid: int) -> str:
        return "\n---\n".join(d[:300] for d in clusters_docs[cid][:5])

    # ── Fallback: no LLM key ─────────────────────────────────────────────────
    if not settings.openai_api_key and not settings.groq_api_key:
        clusters = [
            ClusterSummary(
                cluster_id=cid,
                topic=f"클러스터 {cid}",
                keywords=[],
                doc_count=cluster_sizes[cid],
                is_gap=cid in gap_cluster_ids,
                severity=severities[cid],
                related_clusters=related_map[cid],
                avg_age_days=cluster_avg_age[cid],
            )
            for cid in range(n_clusters)
        ]
        return GapsResponse(
            goal=profile.goal,
            total_chunks=total,
            n_clusters=n_clusters,
            required_areas=None,
            clusters=clusters,
            gaps=[],
            roadmap=None,
        )

    # Groq 우선, 없으면 OpenAI
    if settings.groq_api_key:
        client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
        model = "llama-3.3-70b-versatile"
    else:
        client = OpenAI(api_key=settings.openai_api_key)
        model = "gpt-4o-mini"

    # ── Multi-Agent 오케스트레이터 ────────────────────────────────────────────────
    # 기존: Call 0→1→2 순차 호출 (LLM이 ChromaDB 접근 불가, 추측 기반)
    # 개선: KnowledgeMapAgent → RequirementAgent → GapRecommendAgent
    #        각 Agent가 tool calling으로 ChromaDB 직접 조회 → 검색 기반 판정
    ctx = AgentContext(
        clusters_docs=clusters_docs,
        cluster_sizes=cluster_sizes,
        cluster_avg_age=cluster_avg_age,
        n_clusters=n_clusters,
        goal_ctx=goal_ctx,
    )
    orchestrator = GapOrchestrator(client, model, ctx)
    # asyncio.to_thread: 동기 Groq API 호출이 async 이벤트 루프 블로킹 방지
    import functools
    try:
        knowledge_map, required_areas, gap_data, agent_trace = await asyncio.to_thread(
            functools.partial(
                orchestrator.run,
                gap_cluster_ids=gap_cluster_ids,
                review_cluster_ids=review_cluster_ids,
                severities=severities,
                related_map=related_map,
                strong_cluster_ids=strong_cluster_ids,
                profile=profile,
            )
        )
    except openai.RateLimitError as e:
        import re
        m = re.search(r'try again in (\d+m[\d.]+s)', str(e))
        wait = f" ({m.group(1)} 후 재시도)" if m else ""
        raise HTTPException(status_code=429, detail=f"Groq 일일 토큰 한도 초과{wait}. 내일 다시 시도해주세요.")
    except openai.AuthenticationError:
        raise HTTPException(status_code=401, detail="API 키가 올바르지 않습니다.")
    except openai.APIConnectionError:
        raise HTTPException(status_code=502, detail="LLM 서버에 연결할 수 없습니다.")

    label_map: dict[int, dict] = {item["id"]: item for item in knowledge_map}

    clusters = [
        ClusterSummary(
            cluster_id=cid,
            topic=label_map.get(cid, {}).get("topic", f"클러스터 {cid}"),
            keywords=label_map.get(cid, {}).get("keywords", []),
            doc_count=cluster_sizes[cid],
            is_gap=cid in gap_cluster_ids,
            severity=severities[cid],
            related_clusters=related_map[cid],
            avg_age_days=cluster_avg_age[cid],
        )
        for cid in range(n_clusters)
    ]

    gaps = [
        GapRecommendation(
            area=item.get("area", ""),
            severity=item.get("severity", "medium"),
            gap_type=item.get("gap_type", "missing"),
            goal_relevance=item.get("goal_relevance", "medium"),
            related_strong_topic=(
                ", ".join(item["related_strong_topic"])
                if isinstance(item.get("related_strong_topic"), list)
                else item.get("related_strong_topic")
            ),
            reason=item.get("reason", ""),
            recommendation=item.get("recommendation", ""),
            resources=item.get("resources", []) if isinstance(item.get("resources"), list) else [],
        )
        for item in gap_data.get("gaps", [])
    ]

    roadmap: list[RoadmapPhase] | None = None
    if goal_ctx and gap_data.get("roadmap"):
        roadmap = [
            RoadmapPhase(
                phase=item.get("phase", i + 1),
                period=item.get("period", ""),
                focus=item.get("focus", ""),
                topics=item.get("topics", []),
            )
            for i, item in enumerate(gap_data["roadmap"])
        ]

    return GapsResponse(
        goal=profile.goal,
        total_chunks=total,
        n_clusters=n_clusters,
        required_areas=required_areas if required_areas else None,
        clusters=clusters,
        gaps=gaps,
        roadmap=roadmap,
        agent_trace=agent_trace if agent_trace else None,
    )
