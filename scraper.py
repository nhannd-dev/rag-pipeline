"""Scrape support.optisigns.com via the Zendesk Help Center API and convert articles to Markdown."""

import argparse
import hashlib
import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from markdownify import MarkdownConverter

BASE_URL = "https://support.optisigns.com"
API_URL = f"{BASE_URL}/api/v2/help_center/en-us/articles.json"

log = logging.getLogger(__name__)


@dataclass
class Article:
    id: int
    title: str
    url: str
    updated_at: str
    filename: str
    markdown: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()


class ZendeskConverter(MarkdownConverter):
    """markdownify tweaks for Zendesk article bodies."""

    def convert_a(self, el, text, parent_tags):
        # Keep named anchors (<a name="X"></a>) so in-page "#X" links still resolve.
        if not el.get("href") and (el.get("name") or el.get("id")):
            return f'<a id="{html.escape(el.get("name") or el.get("id"))}"></a>'
        return super().convert_a(el, text, parent_tags)

    def convert_iframe(self, el, text, parent_tags):
        # Embedded videos would otherwise vanish; keep them as plain links.
        src = el.get("src", "")
        if not src:
            return ""
        if src.startswith("//"):
            src = "https:" + src
        label = "YouTube video" if "youtube" in src else "Embedded content"
        return f"\n\n[{label}]({src})\n\n"

    def convert_img(self, el, text, parent_tags):
        # Zendesk often glues images to the following sentence; give them their own block.
        md = super().convert_img(el, text, parent_tags)
        if not md or parent_tags & {"td", "th", "a", "_inline"}:
            return md
        return f"\n\n{md}\n\n"


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "form", "button"]):
        tag.decompose()
    # Site-relative URLs become absolute; in-page anchors (#...) and mailto: stay untouched.
    for tag, attr in (("a", "href"), ("img", "src")):
        for el in soup.find_all(tag, **{attr: True}):
            if el[attr].startswith("/") and not el[attr].startswith("//"):
                el[attr] = BASE_URL + el[attr]
    headings = soup.find_all(re.compile(r"^h[1-6]$"))
    # Headings are already emphasized; bold inside them just adds noise.
    for h in headings:
        for b in h.find_all(["strong", "b"]):
            b.unwrap()
    # The file's H1 is the article title, so demote body headings if any use H1.
    if any(h.name == "h1" for h in headings):
        for h in headings:
            h.name = f"h{min(int(h.name[1]) + 1, 6)}"
    return str(soup)


def html_to_markdown(html: str) -> str:
    md = ZendeskConverter(
        heading_style="ATX",
        bullets="-",
        code_language="",
        table_infer_header=True,  # Zendesk tables rarely use <thead>
        # Source text contains literal ***ALL***, VIDEO_ID, etc.; escaping them only adds backslashes.
        escape_asterisks=False,
        escape_underscores=False,
        strip=["span", "u", "colgroup", "col"],
    ).convert(clean_html(html))
    md = re.sub(r"[ \t]+\n", "\n", md)  # trailing whitespace
    md = re.sub(r"\n{3,}", "\n\n", md)  # collapse blank lines
    return md.strip() + "\n"


def build_markdown(raw: dict) -> str:
    header = (
        f"# {raw['title'].strip()}\n\n"
        f"Article URL: {raw['html_url']}\n"
        f"Last updated: {raw['updated_at']}\n\n"
    )
    return header + html_to_markdown(raw.get("body") or "")


def iter_raw_articles(session: requests.Session, limit: int | None = None):
    url, seen = f"{API_URL}?per_page=100&sort_by=created_at&sort_order=asc", 0
    while url:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        for raw in data["articles"]:
            if raw.get("draft"):
                continue
            yield raw
            seen += 1
            if limit and seen >= limit:
                return
        url = data.get("next_page")


def fetch_articles(limit: int | None = None) -> list[Article]:
    session = requests.Session()
    session.headers["User-Agent"] = "rag-pipeline-scraper/1.0"
    articles = []
    for raw in iter_raw_articles(session, limit):
        articles.append(
            Article(
                id=raw["id"],
                title=raw["title"].strip(),
                url=raw["html_url"],
                updated_at=raw["updated_at"],
                # id prefix keeps filenames unique and stable even if two titles collide.
                filename=f"{raw['id']}-{slugify(raw['title'])}.md",
                markdown=build_markdown(raw),
            )
        )
    log.info("Fetched %d articles", len(articles))
    return articles


def save_articles(articles: list[Article], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in articles:
        # Force LF so the content hash is identical on Windows and in the Linux container.
        (out_dir / a.filename).write_text(a.markdown, encoding="utf-8", newline="\n")
    log.info("Wrote %d Markdown files to %s", len(articles), out_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("articles"))
    parser.add_argument("--limit", type=int, default=None, help="max articles (default: all)")
    args = parser.parse_args()
    save_articles(fetch_articles(args.limit), args.out)
