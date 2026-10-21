FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt \
    # Bake tiktoken's encoding into the image so runs don't depend on downloading it.
    && python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

COPY scraper.py uploader.py main.py ./

RUN useradd --create-home app && chown app /app
USER app

# `docker run -e API_KEY=... <image> main.py` runs the job once and exits.
ENTRYPOINT ["python"]
CMD ["main.py"]
