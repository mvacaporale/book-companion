# Book Companion MCP Server - Cloud Run Deployment
FROM python:3.11-slim

# Install uv for fast dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml uv.lock* ./
COPY book_companion/ ./book_companion/

# Install dependencies
RUN uv sync --frozen --no-dev

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8080
# Data directory - can be overridden or mounted
ENV BOOKRC_DB_PATH=/data/bookrc

# Expose port
EXPOSE 8080

# Run MCP server with SSE transport
# Cloud Run sets PORT env var, we use 8080 by default
CMD ["uv", "run", "python", "-m", "book_companion.mcp.server", "sse"]
