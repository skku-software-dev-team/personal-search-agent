from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sklearn.cluster import KMeans
import numpy as np
from openai import OpenAI

from config import settings
from db import get_collection

router = APIRouter()


# ── Response models ───────────────────────────────────────────────────────────

class ClusterSummary(BaseModel):
    cluster_id: int
    topic: str
    keywords: list[str]
    doc_count: int
    is_gap: bool


class GapRecommendation(BaseModel):
    area: str
    reason: str
    recommendation: str


class GapsResponse(BaseModel):
    total_chunks: int
    n_clusters: int
    clusters: list[ClusterSummary]
    gaps: list[GapRecommendation]


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

    # Determine gap clusters (significantly smaller than average)
    gap_cluster_ids = {
        cid for cid, size in cluster_sizes.items() if size < avg_size * 0.5
    }

    # Build representative text per cluster (first ~300 chars of each doc, up to 5 docs)
    def cluster_sample(cid: int) -> str:
        sample_docs = clusters_docs[cid][:5]
        return "\n---\n".join(d[:300] for d in sample_docs)

    if not settings.openai_api_key:
        # Return clusters without LLM labels when no API key is configured
        clusters = [
            ClusterSummary(
                cluster_id=cid,
                topic=f"클러스터 {cid}",
                keywords=[],
                doc_count=cluster_sizes[cid],
                is_gap=cid in gap_cluster_ids,
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

    import json
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
        )
        for cid in range(n_clusters)
    ]

    # Call 2: generate recommendations for gap clusters only
    gaps: list[GapRecommendation] = []
    if gap_cluster_ids:
        gap_descriptions = "\n".join(
            f"- 클러스터 {cid}: 주제={label_map.get(cid, {}).get('topic', '?')}, "
            f"키워드={label_map.get(cid, {}).get('keywords', [])}, 문서 수={cluster_sizes[cid]}"
            for cid in sorted(gap_cluster_ids)
        )
        gap_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "당신은 개인 지식 관리 전문가입니다. "
                        "지식 공백(gap)이 발견된 클러스터에 대해 구체적인 보완 추천을 해주세요. "
                        "반드시 다음 JSON 형식으로만 응답하세요:\n"
                        '{"gaps": [{"area": "공백 영역", "reason": "부족한 이유", "recommendation": "추천 행동"}]}'
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
