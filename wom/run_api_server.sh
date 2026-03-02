#!/bin/bash
# Run WOM API server using uv environment

uv run gunicorn \
  -w 6 \
  --threads=12 \
  --worker-class=gthread \
  -b 0.0.0.0:5000 \
  --timeout 120 \
  --max-requests 500 \
  --max-requests-jitter 100 \
  --graceful-timeout 120 \
  --access-logfile logs/access.log \
  scripts.wom_api:app