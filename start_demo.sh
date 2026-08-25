#!/usr/bin/env bash
# Local-only, screen-share friendly interview demo launcher.
set -euo pipefail

demo_root="$(cd "$(dirname "$0")" && pwd)"
cd "$demo_root"

if [[ ! -f .env ]]; then
  cp .env.example .env
  echo "Created .env from .env.example. Add OPENAI_API_KEY for DeepSeek generation; retrieval demo works without it."
fi

set -a
source .env
set +a

exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000
