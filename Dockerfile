# ─────────────────────────────────────────────────────────────
# Medical Triage Environment — Dockerfile
# ─────────────────────────────────────────────────────────────
# WHY THESE CHOICES:
#   - python:3.12-slim → minimal image, fast build, 2 vCPU compatible
#   - WORKDIR /app → standard convention, matches uvicorn CMD
#   - No GPU required → all graders are pure Python logic
#   - HEALTHCHECK → required by HF Spaces automated validation
#   - app_port=8000 matches openenv.yaml
# ─────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# Install system dependencies (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (Docker layer caching)
COPY server/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Copy all source files
COPY __init__.py /app/__init__.py
COPY models.py /app/models.py
COPY client.py /app/client.py
COPY server/ /app/server/

# Load environment variables from .env file if it exists
# RUN if [ -f .env ]; then export $(cat .env | xargs); fi

# Health check — HF Spaces pings /health every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose port declared in openenv.yaml
EXPOSE 7860

# Environment variables (can be overridden at runtime)
ENV API_BASE_URL=""
ENV MODEL_NAME=""
ENV HF_TOKEN=""
ENV ENABLE_WEB_INTERFACE=true

# Run the FastAPI server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
