# TODO: Finalize reproducible offline container image (Python 3.11, 2 CPU cores, 4 GB RAM, no network)
FROM python:3.11-slim

WORKDIR /app

# TODO: Copy dependency metadata and install offline wheels / pinned dependencies
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
COPY tests/ ./tests/
COPY configs/ ./configs/
COPY data/ ./data/

# TODO: Run pip install --no-cache-dir -e . and configure non-root runner user
RUN pip install --no-cache-dir -e .

# TODO: Set default verification entrypoint command
CMD ["python", "-m", "finaudit_env.verify_all"]
