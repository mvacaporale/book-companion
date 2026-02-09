# Book Companion MCP Server - Cloud Run Deployment
FROM python:3.11-slim

# Install system dependencies for PDF parsing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy all project files
COPY pyproject.toml uv.lock README.md ./
COPY book_companion/ ./book_companion/

# Create virtual environment and install dependencies
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install -e .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
# Data directory - can be overridden or mounted
ENV BOOKRC_DB_PATH=/data/bookrc

# Expose port
EXPOSE 8080

# Run MCP server with HTTP transport
CMD ["python", "-m", "book_companion.mcp.server", "http"]
