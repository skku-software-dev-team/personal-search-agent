from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from db import get_collection
import json


@tool
def get_documents_by_period(start_date: str, end_date: str) -> str:
    """
    특정 기간의 문서를 ChromaDB에서 가져옵니다.
    start_date, end_date 형식: YYYY-MM-DD
    """
    # 날짜를 int로 변환 (예: "2025-06-20" → 20250620)
    start_int = int(start_date.replace("-", ""))
    end_int = int(end_date.replace("-", ""))

    collection = get_collection()
    results = collection.get(
        where={
            "$and": [
                {"created_at": {"$gte": start_int}},
                {"created_at": {"$lte": end_int}},
            ]
        },
        include=["documents", "metadatas"],
    )
    if not results["documents"]:
        return f"{start_date} ~ {end_date} 기간에 문서 없음"

    docs = []
    for doc, meta in zip(results["documents"], results["metadatas"]):
        docs.append({
            "title": meta.get("title", "제목없음"),
            "date": meta.get("created_at", "날짜없음"),
            "content": doc[:300],
        })
    return json.dumps(docs, ensure_ascii=False)

@tool
def get_date_range() -> str:
    """
    ChromaDB에 저장된 문서들의 전체 날짜 범위를 반환합니다.
    """
    collection = get_collection()
    results = collection.get(include=["metadatas"])
    dates = [
        m.get("created_at")
        for m in results["metadatas"]
        if m.get("created_at")
    ]
    if not dates:
        return "문서 없음"
    return json.dumps({
        "earliest": min(dates),
        "latest": max(dates),
        "total_docs": len(dates),
    }, ensure_ascii=False)


def build_timeline_agent() -> AgentExecutor:
    llm = ChatOpenAI(model="gpt-4o-2024-08-06", temperature=0)

    tools = [get_date_range, get_documents_by_period]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    당신은 사용자의 문서를 분석해서 지적 성장 타임라인을 만드는 AI입니다.

    순서:
    1. get_documents_by_period 툴로 전체 기간 문서 가져오기
    2. 문서 내용 기반으로 의미있는 시기별로 자유롭게 나누기 (개수 제한 없음)
    3. 반드시 아래 JSON 형식으로만 응답 (다른 말 금지)

    {{
    "timeline": [
        {{"keyword": "핵심키워드", "summary": "한줄요약"}},
        {{"keyword": "핵심키워드", "summary": "한줄요약"}}
    ]
    }}
    """),
        MessagesPlaceholder("agent_scratchpad"),
        ("human", "{input}"),
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


async def generate_timeline(start_date: str, end_date: str) -> dict:
    agent = build_timeline_agent()
    result = await agent.ainvoke({
        "input": f"{start_date} 부터 {end_date} 까지의 문서를 분석해서 지적 성장 타임라인을 만들어줘"
    })
    
    print("=== Agent 응답 ===")
    print(result["output"])
    print("=================")
    
    import re, json
    output = result["output"]
    match = re.search(r'\{.*\}', output, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"timeline": []}