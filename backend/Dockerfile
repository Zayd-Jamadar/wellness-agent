FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies first (better layer caching) using only the lockfiles.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the application source and remaining runtime assets.
COPY src ./src
COPY prompts ./prompts
COPY data ./data

# Install the project itself now that the source is present.
RUN uv sync --frozen --no-dev

EXPOSE 8000

CMD ["uv", "run", "wellness", "serve", "--host", "0.0.0.0", "--port", "8000"]
