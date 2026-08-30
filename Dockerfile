FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Accept GH_PAT to clone private repositories if needed
ARG GH_PAT
RUN if [ -n "$GH_PAT" ]; then \
      git config --global url."https://x-access-token:${GH_PAT}@github.com/".insteadOf "https://github.com/"; \
    fi

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml .
COPY . .
RUN uv pip install .

FROM python:3.12-slim

WORKDIR /app
COPY --from=builder /opt/venv /opt/venv
COPY . .
ENV PATH="/opt/venv/bin:$PATH"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
