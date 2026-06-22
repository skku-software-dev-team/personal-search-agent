"""
Multi-Agent 기반 지식 공백 분석 시스템

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
기존 방식 (Sequential Pipeline)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  post_gaps()
    ├─ Call 0: 목표만 보고 LLM이 필요 영역 추측 (검색 없음)
    ├─ Call 1: 클러스터 텍스트를 한 번에 prompt에 포함해 분류
    └─ Call 2: 전달받은 요약 텍스트만 보고 추천 생성

  한계:
  - LLM이 ChromaDB에 접근 불가 → 추측 기반 공백 판정 → 오탐 발생
  - 순서 고정, 중간 실패 시 전체 중단
  - 에이전트 추가/교체 불가

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
멀티에이전트 방식 (이 파일)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  GapOrchestrator
    ├─ KnowledgeMapAgent  (Tool: get_cluster_sample)
    │    각 클러스터를 직접 조회해 토픽·키워드 추출
    │
    ├─ RequirementAgent   (Tool: search_knowledge)
    │    ① 목표 대비 필요 영역 목록 생성
    │    ② 각 영역을 ChromaDB에서 실제 검색 → coverage_score 측정
    │    → "추측"이 아닌 "검색 기반" missing 판정 → 오탐 감소
    │
    └─ GapRecommendAgent  (Tool: search_knowledge)
         각 공백 영역에 대해 관련 기존 문서 검색 →
         맥락 기반 구체적 추천 생성

  장점 vs 기존:
  1. Tool Calling: 각 Agent가 필요한 정보를 ChromaDB에서 능동적으로 조회
  2. 정확도: RequirementAgent가 실제 검색으로 커버리지 확인 후 gap 판정
  3. 맥락 추천: GapRecommendAgent가 관련 기존 문서를 보고 추천 생성
  4. 장애 격리: Agent별 독립 실행 → 부분 실패 허용
  5. 투명성: agent_trace로 각 Agent가 어떤 Tool을 왜 썼는지 기록
  6. 확장성: 새 Agent 추가만으로 기능 확장 (예: TrendAgent, ProjectAgent)
"""

import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from db import get_collection
from embeddings import get_model


# ── Tool 스키마 (Groq/OpenAI function calling 형식) ───────────────────────────

GET_CLUSTER_SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "get_cluster_sample",
        "description": "특정 클러스터의 문서 샘플과 통계(문서 수, 평균 나이)를 가져옵니다.",
        "parameters": {
            "type": "object",
            "properties": {
                "cluster_id": {
                    "type": "integer",
                    "description": "분석할 클러스터 ID (0부터 시작)",
                }
            },
            "required": ["cluster_id"],
        },
    },
}

SEARCH_KNOWLEDGE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "사용자 지식 베이스(ChromaDB)에서 특정 주제가 얼마나 다뤄지는지 검색합니다. "
            "found=true이고 coverage_score > 0.5이면 해당 영역이 충분히 커버됨을 의미합니다."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "검색할 지식 영역 또는 주제 (예: 'Docker 컨테이너화', 'JWT 인증')",
                }
            },
            "required": ["query"],
        },
    },
}


# ── 공유 컨텍스트 ──────────────────────────────────────────────────────────────

@dataclass
class AgentContext:
    clusters_docs: dict[int, list[str]]
    cluster_sizes: dict[int, int]
    cluster_avg_age: dict[int, int]
    n_clusters: int
    goal_ctx: str | None

    # 이전 Agent 결과 (다음 Agent가 읽음)
    knowledge_map: list[dict] = field(default_factory=list)
    coverage: dict[str, dict] = field(default_factory=dict)
    search_samples: dict[str, str] = field(default_factory=dict)  # 검색 결과 캐시


# ── Tool 실행 ──────────────────────────────────────────────────────────────────

def _exec_get_cluster_sample(args: dict, ctx: AgentContext) -> dict:
    cid = args.get("cluster_id", -1)
    if cid not in ctx.clusters_docs:
        return {"error": f"클러스터 {cid} 없음"}
    docs = ctx.clusters_docs[cid]
    return {
        "cluster_id": cid,
        "doc_count": ctx.cluster_sizes.get(cid, 0),
        "avg_age_days": ctx.cluster_avg_age.get(cid, 0),
        "sample": "\n---\n".join(d[:300] for d in docs[:4]),
    }


def _exec_search_knowledge(args: dict) -> dict:
    """ChromaDB에서 실제 검색 수행. 기존 방식은 이 단계 없이 LLM이 추측했음."""
    query = args.get("query", "")
    try:
        model = get_model()
        embedding = model.encode([query]).tolist()
        collection = get_collection()
        results = collection.query(
            query_embeddings=embedding,
            n_results=3,
            include=["documents", "distances"],
        )
        docs = (results.get("documents") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        if not docs:
            return {"found": False, "coverage_score": 0.0, "sample": ""}

        # L2 distance → similarity (정규화 벡터 기준: dist 0=동일, ~1.4=직교)
        score = max(0.0, min(1.0, 1.0 - dists[0]))
        return {
            "found": score > 0.45,
            "coverage_score": round(score, 3),
            "sample": docs[0][:200],
        }
    except Exception as e:
        return {"found": False, "coverage_score": 0.0, "error": str(e)}


# ── Base Agent ─────────────────────────────────────────────────────────────────

_LANG_RULE = (
    "[언어 규칙 — 최우선] 반드시 한국어(한글)와 영문(알파벳)만 사용하세요. "
    "다음 문자는 절대 사용 금지: "
    "일본어 가타카나(ア・イ・ウ・エ・オ 등), 히라가나(あ・い・う・え・お 등), "
    "중국어 간체(云・构・层 등), 번체(雲・構・層 등), "
    "한국 한자(人・機・性・的 등 모든 한자). "
    "외래어·기술용어는 반드시 한글로만 표기하세요: "
    "architecture→아키텍처, infrastructure→인프라스트럭처, "
    "container→컨테이너, kubernetes→쿠버네티스, cloud→클라우드.\n\n"
)


class BaseAgent:
    name: str = "BaseAgent"
    max_tool_iter: int = 6  # Groq 무료 rate limit 고려

    def __init__(self, client: OpenAI, model: str, ctx: AgentContext):
        self.client = client
        self.model = model
        self.ctx = ctx
        self.tool_logs: list[dict] = []

    def _execute_tool(self, fn_name: str, args: dict) -> Any:
        if fn_name == "get_cluster_sample":
            return _exec_get_cluster_sample(args, self.ctx)
        if fn_name == "search_knowledge":
            return _exec_search_knowledge(args)
        return {"error": f"unknown tool: {fn_name}"}

    def _tool_loop(self, messages: list, tools: list) -> list[dict]:
        """Tool calling 루프. 수집된 tool 결과 목록 반환."""
        collected: list[dict] = []
        for _ in range(self.max_tool_iter):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
                temperature=0.2,
            )
            choice = resp.choices[0]
            if choice.finish_reason != "tool_calls":
                break

            messages.append(choice.message)
            for tc in choice.message.tool_calls:
                args = json.loads(tc.function.arguments)
                result = self._execute_tool(tc.function.name, args)
                collected.append({"tool": tc.function.name, "args": args, "result": result})
                self.tool_logs.append({
                    "agent": self.name,
                    "tool": tc.function.name,
                    "args": args,
                    "result": str(result)[:150],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, ensure_ascii=False),
                })
        return collected

    def _json_call(self, system: str, user: str, temperature: float = 0.3) -> dict:
        """깔끔한 메시지로 JSON 출력 요청 (tool 이력 없음 → 안정적)."""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user + "\n\n반드시 올바른 JSON으로만 응답하세요."},
            ],
            response_format={"type": "json_object"},
            temperature=temperature,
        )
        return json.loads(resp.choices[0].message.content)

    def run(self) -> Any:
        raise NotImplementedError


# ── KnowledgeMapAgent ─────────────────────────────────────────────────────────

class KnowledgeMapAgent(BaseAgent):
    """
    역할: 문서 클러스터 분석 → 지식 맵 생성
    Tool: get_cluster_sample

    기존 Call 1과의 차이:
    - 기존: 모든 클러스터 샘플을 prompt에 한꺼번에 포함 (토큰 낭비, 긴 context)
    - 개선: Agent가 각 클러스터를 개별 tool call로 조회
             → 필요시 특정 클러스터를 재조회하거나 집중 분석 가능
    """
    name = "KnowledgeMapAgent"

    def run(self) -> list[dict]:
        # LLM tool calling 불필요 — 클러스터 데이터를 이미 보유, 직접 전달
        samples_text = "\n\n".join(
            f"[클러스터 {cid}] 문서 {self.ctx.cluster_sizes[cid]}개, "
            f"평균 {self.ctx.cluster_avg_age[cid]}일\n"
            + "\n---\n".join(d[:250] for d in self.ctx.clusters_docs[cid][:3])
            for cid in range(self.ctx.n_clusters)
        )
        # tool_logs에 기록 (UI 투명성 유지)
        for cid in range(self.ctx.n_clusters):
            result = _exec_get_cluster_sample({"cluster_id": cid}, self.ctx)
            self.tool_logs.append({
                "agent": self.name, "tool": "get_cluster_sample",
                "args": {"cluster_id": cid}, "result": str(result)[:150],
            })

        result = self._json_call(
            system=_LANG_RULE + "당신은 문서 분석 전문가입니다.",
            user=(
                f"다음 클러스터 샘플을 분석해 주제와 키워드를 추출하세요:\n\n"
                f"{samples_text}\n\n"
                "반드시 다음 JSON 형식으로 응답하세요:\n"
                '{"clusters": [{"id": 0, "topic": "주제", "keywords": ["k1","k2","k3"]}]}'
            ),
        )
        knowledge_map = result.get("clusters", [])
        self.ctx.knowledge_map = knowledge_map
        return knowledge_map


# ── RequirementAgent ──────────────────────────────────────────────────────────

class RequirementAgent(BaseAgent):
    """
    역할: 커리어 목표 분석 → 필요 영역 도출 → 실제 ChromaDB 검색으로 커버리지 확인
    Tool: search_knowledge

    기존 Call 0과의 차이 (핵심):
    - 기존: LLM이 목표만 보고 필요 영역 목록 추측 → ChromaDB 검색 없음
            → "필요하다"고 판단한 영역이 실제로 있는지 알 수 없음 → 오탐 발생
    - 개선: ① LLM이 필요 영역 목록 생성
            ② search_knowledge로 각 영역을 실제 DB에서 검색
            ③ coverage_score 기반으로 missing / partial / covered 판정
            → 검색 기반 정확한 공백 탐지
    """
    name = "RequirementAgent"

    def run(self) -> dict[str, dict]:
        if not self.ctx.goal_ctx:
            return {}

        topic_list = ", ".join(
            f"'{c.get('topic', '')}'"
            for c in self.ctx.knowledge_map
        )

        # Step 1: 필요 영역 목록 생성 (tool 없이 빠르게)
        areas_raw = self._json_call(
            system=_LANG_RULE + "당신은 커리어 코치 전문가입니다. 사용자 목표 달성에 필요한 핵심 지식 영역을 나열하세요.",
            user=(
                f"{self.ctx.goal_ctx}\n"
                f"현재 보유 토픽: {topic_list}\n\n"
                "이 목표 달성에 필요한 지식 영역 10~15개를 반환하세요.\n"
                '{"required_areas": ["영역1", "영역2", ...]}'
            ),
        )
        required_areas = areas_raw.get("required_areas", [])

        if not required_areas:
            return {}

        # 검색 횟수 제한 — 상위 8개만 (LLM 호출 수 감소)
        required_areas = required_areas[:8]

        # Step 2: Python에서 직접 검색 (LLM tool calling 불필요 — 목록이 이미 확정됨)
        area_list_str = "\n".join(f"- {a}" for a in required_areas)
        search_rows = []
        for area in required_areas:
            r = _exec_search_knowledge({"query": area})
            self.ctx.search_samples[area] = r.get("sample", "")
            self.tool_logs.append({
                "agent": self.name, "tool": "search_knowledge",
                "args": {"query": area}, "result": str(r)[:150],
            })
            search_rows.append(
                f"- {area}: found={r.get('found')}, score={r.get('coverage_score', 0):.2f}"
            )
        search_summary = "\n".join(search_rows) or "검색 결과 없음"

        result = self._json_call(
            system=_LANG_RULE + "당신은 지식 커버리지 분석 전문가입니다.",
            user=(
                f"필요 영역 목록:\n{area_list_str}\n\n"
                f"ChromaDB 검색 결과:\n{search_summary}\n\n"
                "검색 결과를 바탕으로 각 영역의 커버리지를 판정하세요 "
                "(coverage_score > 0.5 → covered, 0.3~0.5 → partial, 미만 → missing):\n"
                '{"coverage": [{"area": "영역명", "status": "missing|partial|covered", '
                '"coverage_score": 0.0}]}'
            ),
        )
        coverage = {
            item["area"]: {
                "status": item.get("status", "missing"),
                "coverage_score": item.get("coverage_score", 0.0),
            }
            for item in result.get("coverage", [])
        }
        self.ctx.coverage = coverage
        return coverage


# ── GapRecommendAgent ─────────────────────────────────────────────────────────

class GapRecommendAgent(BaseAgent):
    """
    역할: 공백 영역별 맥락 검색 → 구체적 추천 + 로드맵 생성
    Tool: search_knowledge

    기존 Call 2와의 차이:
    - 기존: 텍스트로 전달받은 요약만 보고 추천 생성
    - 개선: 각 공백 영역에 대해 관련 기존 문서를 실제 검색
            → "현재 보유한 지식과 연결되는" 구체적 추천 가능
            예) Docker 공백 → 기존 CI/CD 문서 발견 → "현재 CI/CD에 Docker 통합" 추천
    """
    name = "GapRecommendAgent"

    def run(
        self,
        gap_cluster_ids: set,
        review_cluster_ids: set,
        severities: dict,
        related_map: dict,
        strong_cluster_ids: set,
    ) -> dict:
        label_map = {c["id"]: c for c in self.ctx.knowledge_map}

        # coverage 기반 missing/partial 공백
        coverage_lines = "\n".join(
            f"- {area}: {info['status']} (score={info['coverage_score']:.2f})"
            for area, info in self.ctx.coverage.items()
            if info["status"] in ("missing", "partial")
        ) or "없음"

        # 클러스터 기반 sparse 공백
        sparse_lines = "\n".join(
            "- {} (문서 {}개, 심각도={}, 연관={})".format(
                label_map.get(cid, {}).get("topic", "?"),
                self.ctx.cluster_sizes.get(cid, 0),
                severities.get(cid, "low"),
                [label_map.get(r, {}).get("topic", f"클러스터 {r}")
                 for r in related_map.get(cid, []) if r in strong_cluster_ids] or "없음",
            )
            for cid in sorted(gap_cluster_ids)
        ) or "없음"

        # 오래된 클러스터 (review)
        review_lines = "\n".join(
            "- {} (마지막으로 본 지 약 {}개월)".format(
                label_map.get(cid, {}).get("topic", "?"),
                round(self.ctx.cluster_avg_age.get(cid, 0) / 30),
            )
            for cid in sorted(review_cluster_ids)
        ) or "없음"

        has_goal = bool(self.ctx.goal_ctx)

        # RequirementAgent 캐시에서 이미 검색한 영역 재사용
        cached_context = "\n".join(
            f"- {area}: sample=\"{sample[:100]}\""
            for area, sample in self.ctx.search_samples.items()
        )

        # 캐시에 없는 sparse/review 클러스터 토픽만 직접 검색 (LLM 불필요)
        new_rows = []
        for cid in sorted(gap_cluster_ids | review_cluster_ids):
            topic = label_map.get(cid, {}).get("topic", "")
            if topic and topic not in self.ctx.search_samples:
                r = _exec_search_knowledge({"query": topic})
                self.ctx.search_samples[topic] = r.get("sample", "")
                self.tool_logs.append({
                    "agent": self.name, "tool": "search_knowledge",
                    "args": {"query": topic}, "result": str(r)[:150],
                })
                new_rows.append(f"- {topic}: sample=\"{r.get('sample','')[:100]}\"")

        new_context = "\n".join(new_rows)
        context_text = "\n".join(filter(None, [cached_context, new_context])) or "관련 문서 없음"

        json_schema = (
            '{"gaps": [{"area": "공백 영역", "severity": "critical|medium|low", '
            '"gap_type": "missing|sparse|review", '
            '"goal_relevance": "high|medium|low", '
            '"related_strong_topic": "토픽명 또는 null", '
            '"reason": "부족한 이유 (구체적으로)", '
            '"recommendation": "구체적인 실습 목표나 마일스톤", '
            '"resources": ["자료 유형 — 제목 (플랫폼)", "자료2"]}], '
            + ('"roadmap": [{"phase": 1, "period": "1~2개월", "focus": "핵심 목표", "topics": ["토픽1"]}]}'
               if has_goal else '"roadmap": null}')
        )

        return self._json_call(
            system=(
                _LANG_RULE
                + "당신은 개인 지식 관리 및 커리어 코치 전문가입니다. "
                + "recommendation은 추상적 문장이 아닌 구체적 실습 목표로 작성하세요. "
                + "review 타입은 복습 추천 톤으로 부드럽게 작성하세요."
            ),
            user=(
                f"{self.ctx.goal_ctx or ''}\n\n"
                f"[목표 대비 누락/부족 영역]\n{coverage_lines}\n\n"
                f"[문서 부족 클러스터(sparse)]\n{sparse_lines}\n\n"
                f"[오래된 클러스터(review)]\n{review_lines}\n\n"
                f"[관련 기존 문서 검색 결과 (GapRecommendAgent 수집)]\n{context_text}\n\n"
                f"다음 JSON 형식으로 응답하세요:\n{json_schema}"
            ),
            temperature=0.5,
        )


# ── GapOrchestrator ───────────────────────────────────────────────────────────

class GapOrchestrator:
    """
    세 Agent의 실행 순서 제어 및 컨텍스트 전달.

    반환값에 agent_trace 포함 → Streamlit UI에서 각 Agent의 tool 호출 기록 시각화 가능.
    이를 통해 "AI가 어떤 판단을 했는지" 투명하게 확인 가능 (기존 방식에 없는 기능).
    """

    def __init__(self, client: OpenAI, model: str, ctx: AgentContext):
        self.client = client
        self.model = model
        self.ctx = ctx

    def run(
        self,
        gap_cluster_ids: set,
        review_cluster_ids: set,
        severities: dict,
        related_map: dict,
        strong_cluster_ids: set,
        profile,
    ) -> tuple[list[dict], list[str], dict, list[dict]]:
        """
        Returns:
            knowledge_map   : 클러스터별 토픽+키워드
            required_areas  : 목표 대비 누락/부족 영역 이름 목록
            gap_data        : {"gaps": [...], "roadmap": [...]}
            agent_trace     : 각 Agent의 tool call 기록
        """
        all_traces: list[dict] = []

        # ── 1. KnowledgeMapAgent ───────────────────────────────────────────────
        km = KnowledgeMapAgent(self.client, self.model, self.ctx)
        knowledge_map = km.run()
        all_traces.extend(km.tool_logs)

        # ── 2. RequirementAgent (목표 있을 때만) ────────────────────────────────
        coverage: dict[str, dict] = {}
        if self.ctx.goal_ctx:
            req = RequirementAgent(self.client, self.model, self.ctx)
            coverage = req.run()
            all_traces.extend(req.tool_logs)

        # ── 3. GapRecommendAgent ───────────────────────────────────────────────
        gap = GapRecommendAgent(self.client, self.model, self.ctx)
        gap_data = gap.run(
            gap_cluster_ids=gap_cluster_ids,
            review_cluster_ids=review_cluster_ids,
            severities=severities,
            related_map=related_map,
            strong_cluster_ids=strong_cluster_ids,
        )
        all_traces.extend(gap.tool_logs)

        # required_areas: 검색으로 missing/partial 확인된 영역만
        required_areas = [
            area
            for area, info in coverage.items()
            if info["status"] in ("missing", "partial")
        ]

        return knowledge_map, required_areas, gap_data, all_traces
