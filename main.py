"""Daily job: re-scrape the help center and push only new/changed/deleted articles to the vector store."""

import json
import logging
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
# The assignment runs the image with `-e API_KEY=...`; the OpenAI SDK reads OPENAI_API_KEY.
if not os.getenv("OPENAI_API_KEY") and os.getenv("API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]

import scraper  # noqa: E402
import uploader  # noqa: E402

log = logging.getLogger("main")
SUMMARY_PATH = Path(os.getenv("RUN_SUMMARY_PATH", "last_run.json"))


def run() -> dict:
    started = time.monotonic()
    articles = scraper.fetch_articles()
    # A fresh folder per run, so articles deleted upstream are detected as removed.
    with tempfile.TemporaryDirectory() as tmp:
        scraper.save_articles(articles, Path(tmp))
        result = uploader.sync(Path(tmp))
    return {
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - started, 1),
        "scraped": len(articles),
        "added": len(result.added),
        "updated": len(result.updated),
        "skipped": len(result.skipped),
        "removed": len(result.removed),
        "failed": result.failed,
        "chunks_embedded_est": result.chunks,
        "changed_files": sorted(result.added + result.updated + result.removed),
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2"):
        logging.getLogger(name).setLevel(logging.WARNING)
    if not os.getenv("OPENAI_API_KEY"):
        log.error("Set OPENAI_API_KEY (or API_KEY)")
        return 2
    try:
        summary = run()
    except Exception:
        log.exception("Job failed")
        return 1
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(
        "Run summary: added=%d updated=%d skipped=%d removed=%d failed=%d",
        summary["added"], summary["updated"], summary["skipped"], summary["removed"], summary["failed"],
    )
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
