# =============================================================================
# ZEROFORM — Sparse Matrix Studio
# Production Multi-Platform Dockerfile (Debian Linux + C11 Engine + Gunicorn WSGI)
# =============================================================================

FROM python:3.12-slim-bookworm

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=10000

WORKDIR /app

# Install system build dependencies (GCC compiler and C runtime libraries)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        gcc \
        libc6-dev \
        make && \
    rm -rf /var/lib/apt/lists/*

# Install Python production dependencies
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/backend/requirements.txt

# Copy application components
COPY c_engine/ /app/c_engine/
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY protocol/ /app/protocol/

# Compile the native C11 Linux engine executable from source with strict flags
WORKDIR /app/c_engine
RUN gcc -std=c11 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Werror -O2 \
    zeroform.c dense_matrix.c sparse_matrix.c matrix_operations.c memory_analysis.c src/main.c \
    -o zeroform -lm && \
    chmod 755 zeroform

WORKDIR /app

# Create a non-privileged system user for secure container execution
RUN useradd -m -u 10001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Expose default Render port
EXPOSE 10000

# Launch Flask application using Gunicorn production WSGI server
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 2 --threads 4 --timeout 30 --access-logfile - --error-logfile - 'backend.app:app'"]
