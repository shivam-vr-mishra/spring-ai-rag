#!/usr/bin/env bash
#
# Stop the Spring AI RAG POC.
#
#   ./stop.sh            stop the app (keeps PGVector and your data)
#   ./stop.sh --all      also stop PGVector
#   ./stop.sh --purge    also stop PGVector AND delete all ingested documents
#
set -euo pipefail
cd "$(dirname "$0")"

MODE="app"
case "${1:-}" in
  --all)   MODE="all" ;;
  --purge) MODE="purge" ;;
  -h|--help) sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
  "")      ;;
  *)       echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
esac

PID="$(ss -ltnp 2>/dev/null | grep ':8080 ' | grep -oP 'pid=\K[0-9]+' | head -1 || true)"
if [ -n "$PID" ]; then
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 10); do ss -ltn 2>/dev/null | grep -q ':8080 ' || break; sleep 1; done
  ss -ltn 2>/dev/null | grep -q ':8080 ' && kill -9 "$PID" 2>/dev/null || true
  echo "  ✓ app stopped"
else
  echo "  · app was not running"
fi

case "$MODE" in
  all)   docker compose down >/dev/null 2>&1 && echo "  ✓ PGVector stopped (data kept)" ;;
  purge) docker compose down -v >/dev/null 2>&1 && echo "  ✓ PGVector stopped, all ingested documents deleted" ;;
esac

# Ollama is left running on purpose: it is a shared service and may be in use
# by something else. Stop it yourself with: pkill -f 'ollama serve'
