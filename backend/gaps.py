import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openai
from openai import OpenAI

from config import settings
from db import get_collection
from user_profile import UserProfile, load_profile

router = APIRouter()

RELATED_SIMILARITY_THRESHOLD = 0.6


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


class GapRecommendation(BaseModel):
    area: str
    severity: str             # "low" | "medium" | "critical"
    goal_relevance: str       # "high" | "medium" | "low"
    related_strong_topic: str | None
    reason: str
    recommendation: str


class RoadmapPhase(BaseModel):
    phase: int
    period: str
    focus: str
    topics: list[str]


class GapsResponse(BaseModel):
    goal: str | None
    total_chunks: int
    n_clusters: int
    clusters: list[ClusterSummary]
    gaps: list[GapRecommendation]
    roadmap: list[RoadmapPhase] | None


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


def _merge_profile(request: GapsRequest) -> UserProfile:
    """Request body overrides saved profile field by field."""
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


# ── Route ─────────────────────────────────────────────────────────────────────

@router.post("/gaps", response_model=GapsResponse)
async def post_gaps(body: GapsRequest = GapsRequest()):
    profile = _merge_profile(body)
    goal_ctx = _goal_context(profile)

    collection = get_collection()
    result = collection.get(include=["embeddings", "documents", "metadatas"])

    docs = result.get("documents") or []
    embeddings = result.get("embeddings") or []

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
    for doc, label in zip(docs, labels):
        clusters_docs[int(label)].append(doc)

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

    def cluster_sample(cid: int) -> str:
        return "\n---\n".join(d[:300] for d in clusters_docs[cid][:5])

    # ── Fallback: no OpenAI key ───────────────────────────────────────────────
    if not settings.openai_api_key:
        clusters = [
            ClusterSummary(
                cluster_id=cid,
                topic=f"클러스터 {cid}",
                keywords=[],
                doc_count=cluster_sizes[cid],
                is_gap=cid in gap_cluster_ids,
                severity=severities[cid],
                related_clusters=related_map[cid],
            )
            for cid in range(n_clusters)
        ]
        return GapsResponse(
            goal=profile.goal,
            total_chunks=total,
            n_clusters=n_clusters,
            clusters=clusters,
            gaps=[],
            roadmap=None,
        )

    client = OpenAI(api_key=settings.openai_api_key)

    # ── Call 1: topic + keywords for all clusters ─────────────────────────────
    cluster_prompts = "\n\n".join(
        f"[클러스터 {cid}]\n{cluster_sample(cid)}" for cid in range(n_clusters)
    )
    try:
        label_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 문서 분석 전문가입니다. "
                        "각 클러스터의 주제와 핵심 키워드를 추출해주세요. "
                        "반드시 다음 JSON 형식으로만 응답하세요 (다른 텍스트 없이):\n"
                        '{"clusters": [{"id": 0, "topic": "주제", "keywords": ["k1","k2","k3"]}]}'
                    ),
                },
                {"role": "user", "content": cluster_prompts},
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
        )
    except openai.AuthenticationError:
        raise HTTPException(
            status_code=401,
            detail="OpenAI API 키가 올바르지 않습니다. .env 파일의 OPENAI_API_KEY를 확인해주세요.",
        )
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
    except openai.APIConnectionError:
        raise HTTPException(status_code=502, detail="OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.")
    except openai.OpenAIError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API 오류: {e}")

    label_data = json.loads(label_response.choices[0].message.content)
    label_map: dict[int, dict] = {
        item["id"]: item for item in label_data.get("clusters", [])
    }

    clusters = [
        ClusterSummary(
            cluster_id=cid,
            topic=label_map.get(cid, {}).get("topic", f"클러스터 {cid}"),
            keywords=label_map.get(cid, {}).get("keywords", []),
            doc_count=cluster_sizes[cid],
            is_gap=cid in gap_cluster_ids,
            severity=severities[cid],
            related_clusters=related_map[cid],
        )
        for cid in range(n_clusters)
    ]

    # ── Call 2: gap recommendations + optional roadmap ────────────────────────
    gaps: list[GapRecommendation] = []
    roadmap: list[RoadmapPhase] | None = None

    if gap_cluster_ids:
        gap_lines = []
        for cid in sorted(gap_cluster_ids):
            related_strong = [
                label_map.get(r, {}).get("topic", f"클러스터 {r}")
                for r in related_map[cid]
                if r in strong_cluster_ids
            ]
            gap_lines.append(
                f"- 클러스터 {cid}: 주제={label_map.get(cid, {}).get('topic', '?')}, "
                f"키워드={label_map.get(cid, {}).get('keywords', [])}, "
                f"문서 수={cluster_sizes[cid]}, 심각도={severities[cid]}, "
                f"연관된 강한 토픽={related_strong or '없음'}"
            )
        gap_descriptions = "\n".join(gap_lines)

        if goal_ctx:
            user_content = (
                f"[사용자 정보]\n{goal_ctx}\n\n"
                f"[지식 공백 클러스터]\n{gap_descriptions}\n\n"
                f"위 사용자의 커리어 목표를 기준으로 분석해주세요."
            )
            system_content = (
                "당신은 개인 지식 관리 및 커리어 코치 전문가입니다.\n"
                "사용자의 커리어 목표를 기반으로 지식 공백을 재평가하고 "
                "학습 로드맵을 생성해주세요.\n\n"
                "규칙:\n"
                "1. 목표와 직결된 공백은 severity를 critical로, 무관한 공백은 low로 조정\n"
                "2. goal_relevance는 목표와의 연관성 (high/medium/low)\n"
                "3. roadmap은 목표 기간 안에 달성 가능한 phase별 학습 계획\n\n"
                "반드시 다음 JSON 형식으로만 응답하세요:\n"
                '{"gaps": [{"area": "공백 영역", "severity": "critical|medium|low", '
                '"goal_relevance": "high|medium|low", '
                '"related_strong_topic": "연관 토픽명 또는 null", '
                '"reason": "부족한 이유", "recommendation": "추천 행동"}], '
                '"roadmap": [{"phase": 1, "period": "1~2개월", "focus": "핵심 목표", '
                '"topics": ["토픽1", "토픽2"]}]}'
            )
        else:
            user_content = f"다음 지식 공백 클러스터를 분석해주세요:\n{gap_descriptions}"
            system_content = (
                "당신은 개인 지식 관리 전문가입니다. "
                "지식 공백(gap) 클러스터에 대해 구체적인 보완 추천을 해주세요. "
                "심각도(critical/medium/low)와 연관된 강한 토픽 정보를 활용하세요. "
                "반드시 다음 JSON 형식으로만 응답하세요:\n"
                '{"gaps": [{"area": "공백 영역", "severity": "critical|medium|low", '
                '"goal_relevance": "medium", '
                '"related_strong_topic": "연관 토픽명 또는 null", '
                '"reason": "부족한 이유", "recommendation": "추천 행동"}], '
                '"roadmap": null}'
            )

        try:
            gap_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.5,
            )
        except openai.AuthenticationError:
            raise HTTPException(
                status_code=401,
                detail="OpenAI API 키가 올바르지 않습니다. .env 파일의 OPENAI_API_KEY를 확인해주세요.",
            )
        except openai.RateLimitError:
            raise HTTPException(status_code=429, detail="OpenAI API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
        except openai.APIConnectionError:
            raise HTTPException(status_code=502, detail="OpenAI 서버에 연결할 수 없습니다. 네트워크를 확인해주세요.")
        except openai.OpenAIError as e:
            raise HTTPException(status_code=502, detail=f"OpenAI API 오류: {e}")
        gap_data = json.loads(gap_response.choices[0].message.content)

        gaps = [
            GapRecommendation(
                area=item.get("area", ""),
                severity=item.get("severity", "medium"),
                goal_relevance=item.get("goal_relevance", "medium"),
                related_strong_topic=item.get("related_strong_topic"),
                reason=item.get("reason", ""),
                recommendation=item.get("recommendation", ""),
            )
            for item in gap_data.get("gaps", [])
        ]

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
        clusters=clusters,
        gaps=gaps,
        roadmap=roadmap,
    )
