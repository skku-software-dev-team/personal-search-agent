import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI

from config import settings
from db import get_collection

router = APIRouter()

RELATED_SIMILARITY_THRESHOLD = 0.6


# ── Response models ───────────────────────────────────────────────────────────

class ClusterSummary(BaseModel):
    cluster_id: int
    topic: str
    keywords: list[str]
    doc_count: int
    is_gap: bool
    severity: str           # "none" | "low" | "medium" | "critical"
    related_clusters: list[int]


class GapRecommendation(BaseModel):
    area: str
    severity: str           # "low" | "medium" | "critical"
    related_strong_topic: str | None
    reason: str
    recommendation: str


class GapsResponse(BaseModel):
    total_chunks: int
    n_clusters: int
    clusters: list[ClusterSummary]
    gaps: list[GapRecommendation]


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
    """Returns mapping cid → list of related cluster ids (cosine sim > threshold)."""
    n = len(centers)
    sim = cosine_similarity(centers)
    related: dict[int, list[int]] = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] > RELATED_SIMILARITY_THRESHOLD:
                related[i].append(j)
                related[j].append(i)
    return related


# ── Route ─────────────────────────────────────────────────────────────────────

@router.get("/gaps", response_model=GapsResponse)
async def get_gaps():
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

    # Group docs by cluster
    clusters_docs: dict[int, list[str]] = {i: [] for i in range(n_clusters)}
    for doc, label in zip(docs, labels):
        clusters_docs[int(label)].append(doc)

    cluster_sizes = {cid: len(cdocs) for cid, cdocs in clusters_docs.items()}
    avg_size = total / n_clusters

    # Cluster relationships via cosine similarity of KMeans centers
    related_map = _related_cluster_map(kmeans.cluster_centers_)

    # Strong clusters = above average, used to escalate related gaps to "critical"
    strong_cluster_ids = {cid for cid, sz in cluster_sizes.items() if sz >= avg_size}

    def has_related_strong(cid: int) -> bool:
        return any(r in strong_cluster_ids for r in related_map[cid])

    severities = {
        cid: _gap_severity(cluster_sizes[cid], avg_size, has_related_strong(cid))
        for cid in range(n_clusters)
    }
    gap_cluster_ids = {cid for cid, sev in severities.items() if sev != "none"}

    # Build representative text per cluster (first ~300 chars, up to 5 docs)
    def cluster_sample(cid: int) -> str:
        return "\n---\n".join(d[:300] for d in clusters_docs[cid][:5])

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
            total_chunks=total,
            n_clusters=n_clusters,
            clusters=clusters,
            gaps=[],
        )

    client = OpenAI(api_key=settings.openai_api_key)

    # Call 1: extract topic + keywords for every cluster in one shot
    cluster_prompts = "\n\n".join(
        f"[클러스터 {cid}]\n{cluster_sample(cid)}" for cid in range(n_clusters)
    )
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

    # Call 2: generate recommendations for gap clusters only, with severity context
    gaps: list[GapRecommendation] = []
    if gap_cluster_ids:
        # Build context: for each gap cluster, include its related strong topic name if any
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

        gap_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 개인 지식 관리 전문가입니다. "
                        "지식 공백(gap) 클러스터에 대해 구체적인 보완 추천을 해주세요. "
                        "심각도(critical/medium/low)와 연관된 강한 토픽 정보를 활용하세요. "
                        "반드시 다음 JSON 형식으로만 응답하세요:\n"
                        '{"gaps": [{"area": "공백 영역", "severity": "critical|medium|low", '
                        '"related_strong_topic": "연관 토픽명 또는 null", '
                        '"reason": "부족한 이유", "recommendation": "추천 행동"}]}'
                    ),
                },
                {
                    "role": "user",
                    "content": f"다음 지식 공백 클러스터를 분석해주세요:\n{gap_descriptions}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.5,
        )
        gap_data = json.loads(gap_response.choices[0].message.content)
        gaps = [
            GapRecommendation(
                area=item.get("area", ""),
                severity=item.get("severity", "medium"),
                related_strong_topic=item.get("related_strong_topic"),
                reason=item.get("reason", ""),
                recommendation=item.get("recommendation", ""),
            )
            for item in gap_data.get("gaps", [])
        ]

    return GapsResponse(
        total_chunks=total,
        n_clusters=n_clusters,
        clusters=clusters,
        gaps=gaps,
    )
