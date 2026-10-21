from itertools import count
from types import SimpleNamespace as NS

import pytest

import uploader


class FakeOpenAI:
    """In-memory stand-in for the parts of the OpenAI client the uploader touches."""

    def __init__(self, fail_names=()):
        self.ids = count()
        self.uploaded = {}  # file_id -> filename
        self.store = {}  # vector-store file_id -> NS(id, status, attributes)
        self.fail_names = set(fail_names)
        self.deleted = []
        self.vector_stores = NS(
            list=lambda **kw: [NS(id="vs_1", name=uploader.VECTOR_STORE_NAME)],
            create=lambda **kw: NS(id="vs_new"),
            # Real API: newest first with order="desc".
            files=NS(list=lambda **kw: list(reversed(self.store.values())), delete=self._vs_delete),
            file_batches=NS(create_and_poll=self._batch),
        )
        self.files = NS(create=self._upload, delete=lambda file_id: None)

    def _upload(self, file, purpose):
        file_id = f"file_{next(self.ids)}"
        self.uploaded[file_id] = file[0]
        return NS(id=file_id)

    def _vs_delete(self, file_id, vector_store_id):
        self.deleted.append(self.store.pop(file_id).attributes["filename"])

    def _batch(self, vector_store_id, files):
        failed = 0
        for f in files:
            bad = f["attributes"]["filename"] in self.fail_names
            failed += bad
            self.store[f["file_id"]] = NS(
                id=f["file_id"], status="failed" if bad else "completed", attributes=f["attributes"]
            )
        return NS(id="batch", file_counts=NS(completed=len(files) - failed, failed=failed))


@pytest.fixture
def docs(tmp_path):
    for name in ("a.md", "b.md", "c.md"):
        (tmp_path / name).write_text(f"# {name}\n\nbody", encoding="utf-8")
    return tmp_path


def names(items):
    return sorted(items)


def test_first_run_adds_everything(docs):
    client = FakeOpenAI()
    r = uploader.sync(docs, client)
    assert names(r.added) == ["a.md", "b.md", "c.md"]
    assert (r.updated, r.skipped, r.removed, r.failed) == ([], [], [], 0)
    assert r.chunks == 3


def test_second_run_skips_unchanged(docs):
    client = FakeOpenAI()
    uploader.sync(docs, client)
    r = uploader.sync(docs, client)
    assert names(r.skipped) == ["a.md", "b.md", "c.md"]
    assert r.added == r.updated == [] and r.chunks == 0


def test_changed_file_is_replaced_and_deleted_file_removed(docs):
    client = FakeOpenAI()
    uploader.sync(docs, client)
    (docs / "a.md").write_text("# a.md\n\nnew body", encoding="utf-8")
    (docs / "c.md").unlink()
    (docs / "d.md").write_text("# d", encoding="utf-8")

    r = uploader.sync(docs, client)
    assert (r.added, r.updated, r.skipped, r.removed) == (["d.md"], ["a.md"], ["b.md"], ["c.md"])
    assert names(client.deleted) == ["a.md", "c.md"]
    remaining = names(f.attributes["filename"] for f in client.store.values())
    assert remaining == ["a.md", "b.md", "d.md"]


def test_failed_file_is_retried_next_run(docs):
    client = FakeOpenAI(fail_names={"b.md"})
    assert uploader.sync(docs, client).failed == 1

    client.fail_names.clear()
    r = uploader.sync(docs, client)
    assert r.updated == ["b.md"] and r.failed == 0


def test_old_version_deleted_only_after_new_one_uploaded(docs):
    client = FakeOpenAI()
    uploader.sync(docs, client)
    (docs / "a.md").write_text("changed", encoding="utf-8")
    seen_at_delete = []
    original = client.vector_stores.files.delete
    client.vector_stores.files.delete = lambda file_id, vector_store_id: (
        seen_at_delete.append(sum(f.attributes["filename"] == "a.md" for f in client.store.values())),
        original(file_id=file_id, vector_store_id=vector_store_id),
    )
    uploader.sync(docs, client)
    assert seen_at_delete == [2]  # new copy was already in the store when the old one went


def test_duplicate_copies_from_interrupted_run_are_cleaned_up(docs):
    client = FakeOpenAI()
    uploader.sync(docs, client)
    # Simulate a crash after uploading a new a.md but before deleting the old one.
    client.store["file_dup"] = NS(
        id="file_dup", status="completed",
        attributes={"filename": "a.md", "sha256": uploader.sha256((docs / "a.md").read_bytes())},
    )
    r = uploader.sync(docs, client)
    assert names(r.skipped) == ["a.md", "b.md", "c.md"]
    assert names(f.attributes["filename"] for f in client.store.values()) == ["a.md", "b.md", "c.md"]


def test_empty_folder_refuses_to_wipe_store(tmp_path):
    with pytest.raises(RuntimeError):
        uploader.sync(tmp_path, FakeOpenAI())


@pytest.mark.parametrize("tokens, chunks", [(1, 1), (800, 1), (801, 2), (1400, 2), (1401, 3)])
def test_estimate_chunks_matches_sliding_window(monkeypatch, tokens, chunks):
    monkeypatch.setattr(uploader._encoding, "encode", lambda text: range(tokens), raising=False)
    assert uploader.estimate_chunks("x") == chunks
