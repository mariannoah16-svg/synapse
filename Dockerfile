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

# Install uv (fast package manager — as shown in bootcamp)
RUN pip install --no-cache-dir uv

# Copy dependency files first (layer caching)
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# HuggingFace Spaces uses port 7860
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start server — same command as uv run server
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]