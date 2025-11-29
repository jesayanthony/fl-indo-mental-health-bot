FROM python:3.11-slim

WORKDIR /app

# Install serving dependencies
COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy serving code
COPY serving/ ./serving

# Make /app importable as a module root
ENV PYTHONPATH="/app:${PYTHONPATH}"

# Cloud Run uses PORT env var, default 8080
ENV PORT=8080

CMD ["uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8080"]
