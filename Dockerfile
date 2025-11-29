# Dockerfile at repo root

FROM python:3.11-slim

# System deps (optional, but good to have)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install serving dependencies
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving code
COPY serving/ ./serving

# Environment for model location (matches app.py defaults)
ENV MODEL_BUCKET=mental-health-fl-bot
ENV MODEL_SUBDIR=t5_fedavg_demo
ENV MODEL_LOCAL_DIR=/app/model

# Expose port for Cloud Run (must use 8080)
ENV PORT=8080

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
