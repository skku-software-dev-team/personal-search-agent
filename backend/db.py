import chromadb

_client: chromadb.HttpClient | None = None
_collection_metadata = {"hnsw:space": "cosine"}


def init_collection(
    host: str, port: int, auth_token: str, collection_name: str
) -> None:
    global _client
    _client = chromadb.HttpClient(host=host, port=port)


def get_collection(user_id: str):
    if _client is None:
        raise RuntimeError("ChromaDB client not initialized")
    return _client.get_or_create_collection(
        name=f"user_{user_id}",
        metadata=_collection_metadata,
    )
