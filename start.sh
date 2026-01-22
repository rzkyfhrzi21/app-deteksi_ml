#!/bin/bash
export FLASK_MODE=online
gunicorn api_flask:app \
  --workers 1 \
  --threads 1 \
  --bind 0.0.0.0:$PORT \
  --timeout 120