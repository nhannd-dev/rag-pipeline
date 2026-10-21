"""Sync a folder of Markdown files into an OpenAI vector store, uploading only what changed.

State lives in the vector store itself: each file carries `filename` and `sha256` attributes,
so the job needs no local database and can run statelessly (e.g. in a cron container).
"""

import argparse
import hashlib
import logging
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

VECTOR_STORE_NAME = os.getenv("VECTOR_STORE_NAME", "optibot-kb")
MAX_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 200
CHUNKING_STRATEGY = {
    "type": "static",
    "static": {"max_chunk_size_tokens": MAX_CHUNK_TOKENS, "chunk_overlap_tokens": CHUNK_OVERLAP_TOKENS},
}
BATCH_SIZE = 100

log = logging.getLogger(__name__)
_encoding = tiktoken.get_encoding("o200k_base")


@dataclass
class SyncResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    failed: int = 0
    chunks: int = 0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def estimate_chunks(text: str) -> int:
    """Chunks the static strategy produces: a sliding 800-token window advancing 600 tokens.

    The API does not report chunk counts, so this mirrors its windowing locally.
    """
    n = len(_encoding.encode(text))
    if n <= MAX_CHUNK_TOKENS:
        return 1
    step = MAX_CHUNK_TOKENS - CHUNK_OVERLAP_TOKENS
    return 1 + math.ceil((n - MAX_CHUNK_TOKENS) / step)


def get_or_create_vector_store(client: OpenAI) -> str:
    if vs_id := os.getenv("VECTOR_STORE_ID"):
        return vs_id
    for vs in client.vector_stores.list(limit=100):
        if vs.name == VECTOR_STORE_NAME:
            return vs.id
    vs = client.vector_stores.create(name=VECTOR_STORE_NAME)
    log.info("Created vector store %s (%s)", VECTOR_STORE_NAME, vs.id)
    return vs.id


class RemoteIndex(dict):
    """filename -> {file_id, sha256}; `duplicates` holds extra copies left by an interrupted run."""

    def __init__(self):
        super().__init__()
        self.duplicates: list[str] = []


def list_remote(client: OpenAI, vs_id: str) -> RemoteIndex:
    remote = RemoteIndex()
    for f in client.vector_stores.files.list(vector_store_id=vs_id, limit=100, order="desc"):
        attrs = f.attributes or {}
        if "filename" not in attrs:
            continue
        # A file that failed indexing gets no hash, so the next run re-uploads it.
        entry = {"file_id": f.id, "sha256": attrs.get("sha256") if f.status == "completed" else None}
        name = attrs["filename"]
        if name in remote:
            # Keep the newest copy (list returns newest first by default); drop the rest.
            remote.duplicates.append(f.id)
        else:
            remote[name] = entry
    return remote


def delete_file(client: OpenAI, vs_id: str, file_id: str) -> None:
    client.vector_stores.files.delete(file_id=file_id, vector_store_id=vs_id)
    client.files.delete(file_id)


def upload_files(client: OpenAI, vs_id: str, paths: list[Path]) -> int:
    """Upload files and attach them to the vector store. Returns the number that failed."""

    def upload(path: Path) -> dict:
        data = path.read_bytes()
        file = client.files.create(file=(path.name, data), purpose="assistants")
        return {
            "file_id": file.id,
            "attributes": {"filename": path.name, "sha256": sha256(data)},
            "chunking_strategy": CHUNKING_STRATEGY,
        }

    failed = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        entries = list(pool.map(upload, paths))
    for i in range(0, len(entries), BATCH_SIZE):
        batch = client.vector_stores.file_batches.create_and_poll(
            vector_store_id=vs_id, files=entries[i : i + BATCH_SIZE]
        )
        counts = batch.file_counts
        log.info("Batch %s: %d completed, %d failed", batch.id, counts.completed, counts.failed)
        failed += counts.failed
    return failed


def sync(docs_dir: Path, client: OpenAI | None = None) -> SyncResult:
    local = sorted(docs_dir.glob("*.md"))
    # An empty folder means the scrape broke, not that every article was deleted.
    if not local:
        raise RuntimeError(f"No Markdown files in {docs_dir}; refusing to sync")

    client = client or OpenAI()
    vs_id = get_or_create_vector_store(client)
    remote = list_remote(client, vs_id)
    result = SyncResult()
    to_upload: list[Path] = []
    # Old versions are deleted only after their replacements are uploaded,
    # so a crash mid-run never leaves an article missing from the store.
    to_delete: list[str] = []

    for path in local:
        data = path.read_bytes()
        existing = remote.get(path.name)
        if existing and existing["sha256"] == sha256(data):
            result.skipped.append(path.name)
            continue
        if existing:
            to_delete.append(existing["file_id"])
            result.updated.append(path.name)
        else:
            result.added.append(path.name)
        to_upload.append(path)
        result.chunks += estimate_chunks(data.decode("utf-8"))

    # Articles that disappeared from the help center shouldn't keep answering questions.
    local_names = {p.name for p in local}
    for name, info in remote.items():
        if name not in local_names:
            to_delete.append(info["file_id"])
            result.removed.append(name)

    if to_upload:
        result.failed = upload_files(client, vs_id, to_upload)
    for file_id in to_delete + remote.duplicates:
        delete_file(client, vs_id, file_id)

    log.info(
        "Vector store %s: added=%d updated=%d skipped=%d removed=%d failed=%d | "
        "files embedded=%d, chunks embedded (est.)=%d",
        vs_id, len(result.added), len(result.updated), len(result.skipped), len(result.removed),
        result.failed, len(to_upload) - result.failed, result.chunks,
    )
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=Path("articles"))
    if sync(parser.parse_args().dir).failed:
        raise SystemExit("Some files failed to index; check the batch logs above.")
