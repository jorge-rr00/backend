#!/usr/bin/env bash
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive

apt-get update
apt-get install -y --no-install-recommends \
  tesseract-ocr \
  tesseract-ocr-spa \
  libglib2.0-0 \
  libgl1 \
  libgomp1

rm -rf /var/lib/apt/lists/*

exec gunicorn -b 0.0.0.0:${PORT:-8000} server:app
