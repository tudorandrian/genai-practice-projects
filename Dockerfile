FROM python:3.13-slim
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 app
WORKDIR /app
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-install-project
COPY . .
RUN uv sync --frozen && chown -R app:app /app
# The environment is built as root and only read at run time; the tests run unprivileged.
ENV UV_NO_SYNC=1
USER app
CMD ["uv", "run", "pytest", "-m", "core and not network", "-q"]
