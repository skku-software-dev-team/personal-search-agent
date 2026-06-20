# Personal Knowledge OS

> Google Drive, Notion, 로컬 폴더에 흩어진 내 문서들을 AI로 검색·분석·추천해주는 개인 지식 관리 시스템

---

## 기술 스택

| 역할 | 기술 |
|------|------|
| 문서 처리 + RAG | Langchain |
| 임베딩 | jhgan/ko-sroberta-multitask (HuggingFace) |
| 벡터 DB | ChromaDB |
| API 서버 | FastAPI |
| 프론트엔드 | Streamlit |
| 컨테이너 | Docker + Docker Compose |

---

## 사전 요구사항

- Docker Desktop 설치 및 실행 중
- Python 3.11+
- Git

---

### STEP 1 — 레포 클론 & 진입

```bash
git clone https://github.com/chaerinotcherry/personal-search-agent.git
cd personal-search-agent
```

---

### STEP 2 — 환경변수 파일 설정

```bash
cp .env.example .env
```

`.env` 파일을 열어 아래 값 채우기:

```
OPENAI_API_KEY=sk-xxx
LLM_PROVIDER=openai
EMBEDDING_MODEL=jhgan/ko-sroberta-multitask
CHROMA_AUTH_TOKEN=psa-local-token
LOCAL_FOLDER_PATH=./docs 
```

---

### STEP 3 — 파이썬 가상환경 세팅 (로컬 개발용)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r backend/requirements.txt
```

---

### STEP 4 — Docker로 전체 서비스 실행

```bash
docker compose up --build
```


서비스 주소:

| 서비스 | 주소 |
|--------|------|
| Streamlit UI | http://localhost:8501 |
| FastAPI | http://localhost:8000 |
| FastAPI Docs | http://localhost:8000/docs |
| ChromaDB | http://localhost:8001 |


---

### STEP 6 — ChromaDB 컬렉션 확인

```bash
curl -H "Authorization: Bearer pko-local-token" \
     http://localhost:8001/api/v1/collections
```

---

### STEP 7 — 컨테이너 종료

```bash
docker compose down          # 컨테이너만 종료 (데이터 유지)
docker compose down -v       # 컨테이너 + 벡터 DB 데이터까지 삭제
```

---

## 프로젝트 구조

```
personal-search-agent/
├── backend/
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── frontend/
│   ├── Dockerfile
│   └── app.py
├── sample_docs/
│   └── README.md
├── .env
├── .env.example
├── .gitignore
├── docker-compose.yml
└── README.md
```