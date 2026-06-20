import hashlib
import io
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pydantic import BaseModel

from config import settings
from db import get_collection
from embeddings import get_model

router = APIRouter()

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


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


def ingest_document(text: str, doc_id_key: str, metadata: dict) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    if not chunks:
        return 0

    embeddings = get_model().encode(chunks, show_progress_bar=False).tolist()

    source_hash = hashlib.md5(doc_id_key.encode()).hexdigest()[:8]
    ids = [f"{source_hash}_{i}" for i in range(len(chunks))]
    metadatas = [{**metadata, "chunk_index": i} for i in range(len(chunks))]

    get_collection().upsert(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
    return len(chunks)


# ── Response models ───────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    status: str
    source: str
    filename: str
    chunks_ingested: int
    collection: str


class LocalIngestResponse(BaseModel):
    status: str
    source: str
    files_processed: int
    files_skipped: int
    chunks_ingested: int
    collection: str


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/ingest", response_model=IngestResponse)
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


@router.post("/ingest/local", response_model=LocalIngestResponse)
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
            created_at = int(datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).date().isoformat().replace("-", ""))
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
