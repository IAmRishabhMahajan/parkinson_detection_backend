# Use a lightweight Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies (important for LightGBM and others)
RUN apt-get update && apt-get install -y \
    gcc \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose port
EXPOSE 8000

# Run with Gunicorn + Uvicorn worker
CMD bash -c "gunicorn -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT} api.main:app"

# ENV PORT=8000
# CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "api.main:app"]



