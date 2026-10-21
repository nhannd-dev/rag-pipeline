# rag-pipeline

Scrapes a Zendesk help center into Markdown, loads it into an OpenAI vector store, and keeps it in sync with a daily job. The assistant ("OptiBot") answers from that store via File Search.

![Assistant answering "How do I add a YouTube video?" with citations](docs/screenshot.png)

## Setup

```bash
pip install -r requirements.txt
cp .env.sample .env   # then set OPENAI_API_KEY
```

## Run locally

```bash
python scraper.py              # 1. help center -> articles/*.md
python uploader.py             # 2. articles/ -> vector store (delta only)
python main.py                 # 3. both, as the daily job does (temp dir, writes last_run.json)
pytest -q                      # tests (no network, fake OpenAI client)
```

With Docker (runs once, exits 0):

```bash
docker build -t kb-sync .
docker run --rm -e API_KEY=sk-... kb-sync main.py
```

## How it works

- **Scrape** – `scraper.py` pages through the Zendesk Help Center API (`/api/v2/help_center/en-us/articles.json`), so there is no nav/ads to strip; the article body is converted with `markdownify` plus fixes for Zendesk HTML: bold removed from headings, body H1s demoted under the title, named anchors kept so `#links` resolve, iframes turned into links, tables get a header row, images on their own line, no needless `\*`/`\_` escaping. Each file is `<id>-<slug>.md` and begins with `# Title` and `Article URL: …` so every chunk set can be cited.
- **Chunking** – static, 800 tokens with 200 overlap. Most articles are ~2k tokens, so one how-to section ("Add a YouTube video") usually fits in a chunk, and the overlap keeps numbered steps attached to their heading. The API doesn't report chunk counts, so the logged count is computed locally with `tiktoken` using the same window.
- **Delta detection** – every vector-store file carries `filename` and `sha256` attributes. Each run compares local hashes against them: new → added, hash changed → old file deleted and re-uploaded, unchanged → skipped, gone from the help center → removed. Files that failed indexing are retried. No local state, so the container can run anywhere. An empty scrape aborts instead of wiping the store.

## Daily job

GitHub Actions ([`.github/workflows/daily-sync.yml`](.github/workflows/daily-sync.yml)) builds the Docker image and runs it at 06:00 UTC daily (also on demand). Each run prints `added / updated / skipped / removed / failed` in the job summary and uploads `last_run.json` + `job.log` as an artefact.

**Job logs:** <https://github.com/nhannd-dev/rag-pipeline/actions/workflows/daily-sync.yml>

Required repo secret: `OPENAI_API_KEY`. Optional repo variable: `VECTOR_STORE_ID`.
