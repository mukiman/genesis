# Reproducible offline container image (Python 3.11, 2 CPU cores, 4 GB RAM, no network/GPU required)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Copy project metadata, source code, configs, pre-built datasets, and tests
COPY pyproject.toml README.md DESIGN.md PROJECT_MAP.md LICENSE ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY data/ ./data/
COPY artifacts/ ./artifacts/

# Install package and pinned dependencies
RUN pip install --no-cache-dir -e . && \
    useradd --create-home --shell /bin/bash auditor && \
    chown -R auditor:auditor /app

USER auditor

# Single-command offline verification entrypoint
CMD ["python", "-m", "finaudit_env.verify_all"]
