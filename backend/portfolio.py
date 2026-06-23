import json
import os
from typing import Optional

from auth.dependencies import get_current_user
from database.models import User
from db import get_collection
from fastapi import APIRouter, Depends
from openai import OpenAI
from pydantic import BaseModel

router = APIRouter(prefix="/portfolio", tags=["portfolio"])

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class PortfolioRequest(BaseModel):
    user_name: Optional[str] = "개발자"
    max_docs: Optional[int] = 50


class ProjectItem(BaseModel):
    date: str  # "2024-03" 형식
    title: str
    description: str
    tech_stack: list[str]
    role: str


class PortfolioResponse(BaseModel):
    headline: str  # "자동화를 구축하는 개발자입니다"
    summary: str  # 3~4줄 소개
    projects: list[ProjectItem]
    skills: list[str]


@router.post("/generate", response_model=PortfolioResponse)
async def generate_portfolio(
    req: PortfolioRequest, current_user: User = Depends(get_current_user)
):
    # 1. ChromaDB에서 문서 수집
    try:
        collection = get_collection(current_user.id)
        results = collection.get(limit=req.max_docs, include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
    except Exception:
        docs, metas = [], []

    # 2. 문서 요약 텍스트 구성
    doc_text = ""
    for doc, meta in zip(docs, metas):
        date = meta.get("date", "") or meta.get("created_at", "")
        source = meta.get("source", "") or meta.get("title", "")
        doc_text += f"[{date}] {source}\n{doc[:500]}\n\n"

    if not doc_text:
        doc_text = "문서 없음"

    # 3. GPT-4o 포트폴리오 생성
    prompt = f"""
다음은 사용자({req.user_name})의 문서/노트 목록입니다:

{doc_text}

위 내용을 분석해서 포트폴리오를 JSON으로 생성해주세요.

반드시 아래 JSON 형식만 출력하세요 (마크다운 코드블록 없이):
{{
  "headline": "나를 한 문장으로 설명 (예: 자동화를 구축하는 개발자입니다)",
  "summary": "3~4줄 자기소개",
  "projects": [
    {{
      "date": "YYYY-MM",
      "title": "프로젝트명",
      "description": "2~3줄 설명",
      "tech_stack": ["Python", "FastAPI"],
      "role": "본인 역할"
    }}
  ],
  "skills": ["Python", "FastAPI", "ChromaDB", ...]
}}

projects는 날짜 내림차순으로 정렬하세요.
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()
    # 혹시 코드블록 감싸진 경우 제거
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    return PortfolioResponse(**data)
