# Dockerfile for testing WOM on Linux
FROM python:3.11-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY api/ ./api/
COPY core/ ./core/
COPY loaders/ ./loaders/
COPY tests/ ./tests/
COPY chebi_source/ ./chebi_source/
COPY exometabolite_data/ ./exometabolite_data/
COPY processed_data/ ./processed_data/

# Sync dependencies and run tests
RUN uv sync --extra dev

# Run tests
CMD ["uv", "run", "pytest", "tests/", "-v"]
