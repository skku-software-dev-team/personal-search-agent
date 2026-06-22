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
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="API 요청 한도를 초과했습니다. 잠시 후 다시 시도해주세요.")
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

    # ── Call 0: 목표 달성에 필요한 지식 영역 목록 생성 (목표 있을 때만) ───────────
    required_areas: list[str] = []
    if goal_ctx:
        req_data = _llm_call(
            client,
            model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "[언어 규칙 — 최우선] 모든 텍스트는 한국어(한글)와 영문(알파벳)만 사용하세요. "
                        "일본어 가타카나(ア・イ・ウ 등)와 히라가나(あ・い・う 등), "
                        "중국어 한자(简体·繁體)를 절대 사용하지 마세요. "
                        "외래어는 한글로 표기하세요 (예: 인프라스트럭처, 컨테이너, 쿠버네티스).\n\n"
                        "당신은 커리어 코치 전문가입니다. "
                        "사용자의 목표를 달성하기 위해 반드시 알아야 할 핵심 지식 영역을 나열해주세요. "
                        "각 영역은 구체적이고 학습 가능한 단위로 적어주세요. "
                        "반드시 다음 JSON 형식으로만 응답하세요:\n"
                        '{"required_areas": ["영역1", "영역2", ...]}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"{goal_ctx}\n\n"
                        "이 목표를 달성하기 위해 반드시 알아야 할 핵심 지식 영역 10~15개를 반환해주세요. "
                        "(예: REST API 설계, 데이터베이스 인덱싱, Docker 컨테이너화)"
                    ),
                },
            ],
            temperature=0.3,
        )
        required_areas = req_data.get("required_areas", [])

    # ── Call 1: topic + keywords for all clusters ─────────────────────────────
    cluster_prompts = "\n\n".join(
        f"[클러스터 {cid}]\n{cluster_sample(cid)}" for cid in range(n_clusters)
    )
    label_data = _llm_call(
        client,
        model,
        messages=[
            {
                "role": "system",
                "content": (
                    "[언어 규칙 — 최우선] 모든 텍스트는 한국어(한글)와 영문(알파벳)만 사용하세요. "
                    "일본어 가타카나(ア・イ・ウ 등)와 히라가나(あ・い・う 등), "
                    "중국어 한자(简体·繁體)를 절대 사용하지 마세요. "
                    "외래어는 한글로 표기하세요 (예: 인프라스트럭처, 컨테이너, 쿠버네티스).\n\n"
                    "당신은 문서 분석 전문가입니다. "
                    "각 클러스터의 주제와 핵심 키워드를 추출해주세요. "
                    "반드시 다음 JSON 형식으로만 응답하세요 (다른 텍스트 없이):\n"
                    '{"clusters": [{"id": 0, "topic": "주제", "keywords": ["k1","k2","k3"]}]}'
                ),
            },
            {"role": "user", "content": cluster_prompts},
        ],
        temperature=0.3,
    )
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
            avg_age_days=cluster_avg_age[cid],
        )
        for cid in range(n_clusters)
    ]

    # ── Call 2: gap recommendations + optional roadmap ────────────────────────
    gaps: list[GapRecommendation] = []
    roadmap: list[RoadmapPhase] | None = None

    should_call_gap_llm = (
        bool(goal_ctx and required_areas)
        or bool(gap_cluster_ids)
        or bool(review_cluster_ids)
    )

    if should_call_gap_llm:
        cluster_summary_lines = "\n".join(
            f"- {label_map.get(cid, {}).get('topic', f'클러스터 {cid}')} "
            f"(문서 {cluster_sizes[cid]}개)"
            for cid in range(n_clusters)
        )

        sparse_lines = []
        for cid in sorted(gap_cluster_ids):
            related_strong = [
                label_map.get(r, {}).get("topic", f"클러스터 {r}")
                for r in related_map[cid]
                if r in strong_cluster_ids
            ]
            sparse_lines.append(
                f"- {label_map.get(cid, {}).get('topic', '?')} "
                f"(문서 {cluster_sizes[cid]}개, 심각도={severities[cid]}, "
                f"연관 강한 토픽={related_strong or '없음'})"
            )
        sparse_descriptions = "\n".join(sparse_lines) if sparse_lines else "없음"

        review_lines = []
        for cid in sorted(review_cluster_ids):
            age_months = round(cluster_avg_age[cid] / 30)
            review_lines.append(
                f"- {label_map.get(cid, {}).get('topic', '?')} "
                f"(마지막으로 본 지 약 {age_months}개월)"
            )
        review_descriptions = "\n".join(review_lines) if review_lines else "없음"

        if goal_ctx:
            required_list = "\n".join(f"- {a}" for a in required_areas) if required_areas else "없음"
            user_content = (
                f"[사용자 정보]\n{goal_ctx}\n\n"
                f"[목표 달성에 필요한 지식 영역]\n{required_list}\n\n"
                f"[현재 보유한 문서 클러스터]\n{cluster_summary_lines}\n\n"
                f"[문서가 부족한 클러스터(sparse)]\n{sparse_descriptions}\n\n"
                f"[오래된 클러스터 — 복습 추천 대상]\n{review_descriptions}"
            )
            system_content = (
                "[언어 규칙 — 최우선] 모든 텍스트는 한국어(한글)와 영문(알파벳)만 사용하세요. "
                "일본어 가타카나(ア・イ・ウ 등)와 히라가나(あ・い・う 등), "
                "중국어 한자(简体·繁體)를 절대 사용하지 마세요. "
                "외래어는 한글로 표기하세요 (예: 인프라스트럭처, 컨테이너, 쿠버네티스).\n\n"
                "당신은 개인 지식 관리 및 커리어 코치 전문가입니다.\n\n"
                "분석 방법:\n"
                "1. '필요한 지식 영역' 목록과 '현재 보유한 클러스터'를 비교하세요.\n"
                "   - 클러스터에 해당 영역이 없음 → gap_type: 'missing'\n"
                "   - 클러스터는 있지만 문서가 부족함 → gap_type: 'sparse'\n"
                "2. '오래된 클러스터'는 gap_type: 'review'로 분류하세요.\n"
                "   - reason: '관련 자료를 마지막으로 본 지 N개월이 지났습니다. 내용을 잊었을 수 있어요.' 톤\n"
                "   - recommendation: '이미 알고 계신다면 넘어가도 좋지만, 핵심 개념을 한 번 훑어보시길 권합니다.' 톤\n"
                "3. 목표와 직결된 공백은 severity: 'critical', 간접 관련은 'medium', 무관하면 'low'\n"
                "4. recommendation 작성 규칙:\n"
                "   - '~를 학습하세요' 같은 추상적 문장 금지\n"
                "   - 구체적인 실습 목표나 마일스톤으로 작성\n"
                "   - 예: 'Express.js로 CRUD API 직접 구현 후 Postman으로 테스트해보기'\n"
                "   - 예: 'JWT 발급·검증 흐름을 코드로 구현하고 Refresh Token 전략까지 적용해보기'\n"
                "5. resources: 해당 영역 학습에 적합한 자료 2~3개\n"
                "   - 한국어 강의·책·공식문서 우선\n"
                "   - 형식: '자료 유형 — 제목 (저자/플랫폼)'\n"
                "   - 예: ['책 — 「HTTP 완벽 가이드」 (데이빗 고울리)', '강의 — 인프런 「모든 개발자를 위한 HTTP 웹 기본 지식」 (김영한)', '문서 — MDN Web Docs: HTTP 개요']\n"
                "6. 목표 기간에 맞는 phase별 학습 로드맵을 생성하세요. review 항목은 '복습' 단계로 표시.\n\n"
                "반드시 다음 JSON 형식으로만 응답하세요:\n"
                '{"gaps": [{"area": "공백 영역", "severity": "critical|medium|low", '
                '"gap_type": "missing|sparse|review", '
                '"goal_relevance": "high|medium|low", '
                '"related_strong_topic": "연관 토픽명 또는 null", '
                '"reason": "부족한 이유 (구체적으로)", '
                '"recommendation": "구체적인 실습 목표나 마일스톤", '
                '"resources": ["자료1", "자료2", "자료3"]}], '
                '"roadmap": [{"phase": 1, "period": "1~2개월", "focus": "핵심 목표", '
                '"topics": ["토픽1", "토픽2"]}]}'
            )
        else:
            user_content = (
                f"[현재 보유한 문서 클러스터]\n{cluster_summary_lines}\n\n"
                f"[문서가 부족한 클러스터(sparse)]\n{sparse_descriptions}\n\n"
                f"[오래된 클러스터 — 복습 추천 대상]\n{review_descriptions}"
            )
            system_content = (
                "[언어 규칙 — 최우선] 모든 텍스트는 한국어(한글)와 영문(알파벳)만 사용하세요. "
                "일본어 가타카나(ア・イ・ウ 등)와 히라가나(あ・い・う 등), "
                "중국어 한자(简体·繁體)를 절대 사용하지 마세요. "
                "외래어는 한글로 표기하세요 (예: 인프라스트럭처, 컨테이너, 쿠버네티스).\n\n"
                "당신은 개인 지식 관리 전문가입니다. "
                "문서가 부족한 클러스터와 오래된 클러스터에 대해 보완·복습 추천을 해주세요.\n"
                "오래된 클러스터는 gap_type: 'review'로, 이미 알고 있을 수 있다는 점을 인정하는 부드러운 톤으로 작성하세요.\n\n"
                "recommendation 작성 규칙:\n"
                "- '~를 학습하세요' 같은 추상적 문장 금지\n"
                "- 구체적인 실습 목표나 마일스톤으로 작성\n"
                "resources: 해당 영역 학습에 적합한 자료 2~3개 (한국어 우선, '자료 유형 — 제목 (플랫폼)' 형식)\n\n"
                "반드시 다음 JSON 형식으로만 응답하세요:\n"
                '{"gaps": [{"area": "공백 영역", "severity": "critical|medium|low", '
                '"gap_type": "sparse|review", '
                '"goal_relevance": "medium", '
                '"related_strong_topic": "연관 토픽명 또는 null", '
                '"reason": "부족한 이유 (구체적으로)", '
                '"recommendation": "구체적인 실습 목표나 마일스톤", '
                '"resources": ["자료1", "자료2"]}], '
                '"roadmap": null}'
            )

        gap_data = _llm_call(
            client,
            model,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            temperature=0.5,
        )

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
    )
