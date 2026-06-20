from contextlib import asynccontextmanager
import hashlib
import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from fastapi import FastAPI, File, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from sentence_transformers import SentenceTransformer

# ── Config ────────────────────────────────────────────────────────────────────

class AppSettings(BaseSettings):
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_auth_token: str = "psa-local-token"
    embedding_model: str = "jhgan/ko-sroberta-multitask"
    local_folder_path: str = "/data/local"
    collection_name: str = "documents"
    chunk_size: int = 512
    chunk_overlap: int = 64

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = AppSettings()

# ── Globals (initialised in lifespan) ────────────────────────────────────────

_embed_model: SentenceTransformer | None = None
_collection = None

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}

# ── Document parsers ──────────────────────────────────────────────────────────

def _parse_pdf(data: bytes) -> str:
    import pypdf
    reader = pypdf.PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _parse_docx(data: bytes) -> str:
    import docx
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        doc = docx.Document(tmp)
        return "\n".join(p.text for p in doc.paragraphs)
    finally:
        os.unlink(tmp)


def _parse_md(data: bytes) -> str:
    import markdown
    from bs4 import BeautifulSoup
    html = markdown.markdown(data.decode("utf-8", errors="replace"))
    return BeautifulSoup(html, "html.parser").get_text()


def _parse_txt(data: bytes) -> str:
    return data.decode("utf-8", errors="replace")


_PARSERS = {
    ".pdf": _parse_pdf,
    ".docx": _parse_docx,
    ".md": _parse_md,
    ".txt": _parse_txt,
}


def extract_text(data: bytes, ext: str) -> str:
    parser = _PARSERS.get(ext.lower())
    if parser is None:
        raise ValueError(f"Unsupported extension: {ext}")
    return parser(data)

# ── Ingestion core ────────────────────────────────────────────────────────────

def _make_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )


def ingest_document(text: str, doc_id_key: str, metadata: dict) -> int:
    splitter = _make_splitter()
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    embeddings = _embed_model.encode(chunks, show_progress_bar=False).tolist()

    source_hash = hashlib.md5(doc_id_key.encode()).hexdigest()[:8]
    ids = [f"{source_hash}_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    _collection.upsert(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(chunks)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _embed_model, _collection

    _embed_model = SentenceTransformer(settings.embedding_model)

    client = chromadb.HttpClient(
        host=settings.chroma_host,
        port=settings.chroma_port,
        settings=ChromaSettings(
            chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
            chroma_client_auth_credentials=settings.chroma_auth_token,
            chroma_client_auth_token_transport_header="AUTHORIZATION",
        ),
    )

    _collection = client.get_or_create_collection(
        name=settings.collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    yield

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Personal Knowledge OS", version="0.1.0", lifespan=lifespan)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


class IngestResponse(BaseModel):
    status: str
    source: str
    filename: str
    chunks_ingested: int
    collection: str


@app.post("/ingest", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )

    data = await file.read()
    try:
        text = extract_text(data, ext)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {e}")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Document is empty after parsing.")

    metadata = {
        "source": "local",
        "file_name": file.filename,
        "file_path": file.filename,
        "created_at": datetime.now(timezone.utc).date().isoformat(),
    }
    n = ingest_document(text, doc_id_key=file.filename, metadata=metadata)
    return IngestResponse(
        status="ok",
        source="upload",
        filename=file.filename,
        chunks_ingested=n,
        collection=settings.collection_name,
    )


class LocalIngestResponse(BaseModel):
    status: str
    source: str
    files_processed: int
    files_skipped: int
    chunks_ingested: int
    collection: str


@app.post("/ingest/local", response_model=LocalIngestResponse)
async def ingest_local():
    folder = Path(settings.local_folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Local folder not found: {folder}")

    files_processed = 0
    files_skipped = 0
    total_chunks = 0

    for path in sorted(folder.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            files_skipped += 1
            continue
        try:
            text = extract_text(path.read_bytes(), path.suffix).strip()
        except Exception:
            files_skipped += 1
            continue

        if not text:
            files_skipped += 1
            continue

        try:
            created_at = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        except Exception:
            created_at = datetime.now(timezone.utc).date().isoformat()

        metadata = {
            "source": "local",
            "file_name": path.name,
            "file_path": str(path),
            "created_at": created_at,
        }
        total_chunks += ingest_document(text, doc_id_key=str(path), metadata=metadata)
        files_processed += 1

    return LocalIngestResponse(
        status="ok",
        source="local",
        files_processed=files_processed,
        files_skipped=files_skipped,
        chunks_ingested=total_chunks,
        collection=settings.collection_name,
    )
