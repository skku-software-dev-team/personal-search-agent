from fastapi import APIRouter, HTTPException
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from config import settings
from search import SearchResult, search_documents

router = APIRouter()

TOP_K = 15


# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    source: str | None = None


class ChatSource(BaseModel):
    file_name: str
    file_path: str
    source: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[ChatSource]


# ── Prompts ────────────────────────────────────────────────────────────────────

_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "사용자의 질문을 벡터 검색에 적합한 간결한 검색 쿼리로 바꿔라. "
        "핵심 키워드 중심으로, 쿼리만 출력하고 다른 설명은 하지 마라.",
    ),
    ("human", "{question}"),
])

_answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "당신은 사용자의 개인 문서를 기반으로 답하는 어시스턴트입니다.\n"
        "아래 [참고 문서] 중 질문과 실제로 관련 있는 내용만 사용해서 답하라.\n"
        '관련 있는 문서가 없으면 "관련 문서를 찾지 못했습니다."라고만 답하라.\n'
        "답변에 사용한 문서는 [1], [2] 형식으로 인용 번호를 붙여라.\n\n"
        "[참고 문서]\n{context}",
    ),
    ("human", "{question}"),
])


def _group_by_file(results: list[SearchResult]) -> dict[str, list[SearchResult]]:
    grouped: dict[str, list[SearchResult]] = {}
    for r in results:
        grouped.setdefault(r.file_name, []).append(r)
    return grouped


def _build_context(grouped: dict[str, list[SearchResult]]) -> str:
    blocks = [
        f"[{i}] {file_name}\n" + "\n".join(c.text for c in chunks)
        for i, (file_name, chunks) in enumerate(grouped.items(), start=1)
    ]
    return "\n\n".join(blocks)


# Groq 우선, 없으면 OpenAI (gaps.py와 동일한 fallback 패턴)
def _get_llm() -> ChatOpenAI:
    if settings.groq_api_key:
        return ChatOpenAI(
            model="llama-3.3-70b-versatile",
            temperature=0,
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=settings.openai_api_key)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=422, detail="질문을 입력해주세요.")

    if not settings.groq_api_key and not settings.openai_api_key:
        raise HTTPException(status_code=503, detail="LLM API 키가 설정되지 않았습니다.")

    llm = _get_llm()

    rewritten = await (_rewrite_prompt | llm).ainvoke({"question": request.question})
    rewritten_query = rewritten.content.strip() or request.question

    results = search_documents(rewritten_query, top_k=TOP_K, source=request.source)
    if not results:
        return ChatResponse(answer="관련 문서를 찾지 못했습니다.", sources=[])

    grouped = _group_by_file(results)
    context = _build_context(grouped)

    answer = await (_answer_prompt | llm).ainvoke({"question": request.question, "context": context})

    sources = [
        ChatSource(file_name=file_name, file_path=chunks[0].file_path, source=chunks[0].source)
        for file_name, chunks in grouped.items()
    ]

    return ChatResponse(answer=answer.content.strip(), sources=sources)
