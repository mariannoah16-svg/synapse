# SYNAPSE v5 — Dockerfile
# At ROOT directory (not /server) — as required by bootcamp
# Compatible with HuggingFace Spaces (port 7860)
# Runs on 2 vCPU + 8GB RAM (no GPU, no heavy ML libraries)

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv (fast package manager)
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY requirements.txt .
COPY uv.lock* ./

# Install Python dependencies via pip (reliable for HF Spaces)
RUN pip install --no-cache-dir -r requirements.txt

# Regenerate uv.lock to ensure it's valid and non-empty
RUN uv lock --no-cache 2>/dev/null || true

# Copy all project files
COPY . .

# Ensure uv.lock exists and is valid after full copy
RUN uv lock --no-cache 2>/dev/null || true

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server — uvicorn with server.app:app
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
