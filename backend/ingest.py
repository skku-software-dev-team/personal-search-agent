import asyncio
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
    import re
    # strip control chars and null bytes that break the Rust tokenizer
    def _clean(s: str) -> str:
        s = s.replace("\x00", "")
        s = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", s)
        return " ".join(s.split())

    raw = splitter.split_text(text[:200_000])  # cap at 200k chars (~400 chunks max)
    chunks = [_clean(c) for c in raw if c and isinstance(c, str)]
    chunks = [c for c in chunks if len(c) > 10]
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


class ExternalIngestResponse(BaseModel):
    status: str
    source: str
    pages_processed: int
    pages_skipped: int
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


@router.post("/ingest/notion", response_model=ExternalIngestResponse)
async def ingest_notion():
    if not settings.notion_api_key:
        raise HTTPException(status_code=503, detail="NOTION_API_KEY가 설정되지 않았습니다.")

    from ingest_notion import fetch_all_pages

    try:
        pages = await asyncio.to_thread(fetch_all_pages, settings.notion_api_key)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Notion API 오류: {e}")

    processed, skipped, total_chunks = 0, 0, 0
    for page in pages:
        text = page["text"].strip()
        if not text:
            skipped += 1
            continue
        metadata = {
            "source": "notion",
            "file_name": page["title"],
            "file_path": page["url"],
            "created_at": int(page["last_edited"].replace("-", "")) if page["last_edited"] else 0,
        }
        total_chunks += ingest_document(text, doc_id_key=page["page_id"], metadata=metadata)
        processed += 1

    return ExternalIngestResponse(
        status="ok",
        source="notion",
        pages_processed=processed,
        pages_skipped=skipped,
        chunks_ingested=total_chunks,
        collection=settings.collection_name,
    )


@router.post("/ingest/gdrive", response_model=ExternalIngestResponse)
async def ingest_gdrive(folder_id: str | None = None):
    from ingest_gdrive import download_file, get_drive_service, get_extension, list_files

    try:
        service = await asyncio.to_thread(
            get_drive_service, settings.gdrive_credentials_path, settings.gdrive_token_path
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Google Drive 인증 오류: {e}")

    try:
        files = await asyncio.to_thread(list_files, service, folder_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Google Drive 파일 목록 오류: {e}")

    MAX_FILE_BYTES = 10 * 1024 * 1024  # 10MB
    processed, skipped, total_chunks = 0, 0, 0
    import logging
    logger = logging.getLogger(__name__)

    for f in files:
        ext = get_extension(f["mimeType"])
        if ext not in SUPPORTED_EXTENSIONS:
            skipped += 1
            continue
        size = int(f.get("size", 0) or 0)
        if size > MAX_FILE_BYTES:
            logger.info(f"[gdrive] skip (too large {size//1024//1024}MB): {f['name']}")
            skipped += 1
            continue
        logger.info(f"[gdrive] processing: {f['name']}")
        try:
            data = await asyncio.to_thread(download_file, service, f["id"], f["mimeType"])
            text = extract_text(data, ext).strip()
        except Exception as e:
            logger.warning(f"[gdrive] failed to parse {f['name']}: {e}")
            skipped += 1
            continue
        if not text:
            skipped += 1
            continue

        modified = f.get("modifiedTime", "")[:10].replace("-", "")
        metadata = {
            "source": "gdrive",
            "file_name": f["name"],
            "file_path": f"gdrive://{f['id']}",
            "created_at": int(modified) if modified else 0,
        }
        try:
            n = ingest_document(text, doc_id_key=f["id"], metadata=metadata)
            total_chunks += n
            processed += 1
            logger.info(f"[gdrive] done: {f['name']} ({n} chunks)")
        except Exception as e:
            logger.warning(f"[gdrive] ingest failed {f['name']}: {e}")
            skipped += 1

    return ExternalIngestResponse(
        status="ok",
        source="gdrive",
        pages_processed=processed,
        pages_skipped=skipped,
        chunks_ingested=total_chunks,
        collection=settings.collection_name,
    )
