# Drug Interaction Checker Agent - container image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Run the FastAPI server
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
