#!/bin/bash
source activate python310
python ./scripts/wom_api.py --host 0.0.0.0 --port 5000 --model-dir trained_models/bpr_consumed
gunicorn -w 6 --threads=12 --worker-class=gthread -b 0.0.0.0:5000 --timeout 120 --max-requests 500 --max-requests-jitter 100 --graceful-timeout 120 scripts.wom_api:app --access-logfile /app/logs/access.log