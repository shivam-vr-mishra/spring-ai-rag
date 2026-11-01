#!/usr/bin/env bash
#
# Start the Spring AI RAG POC.
#
#   ./start.sh                      ollama (default — local, free, no API key)
#   ./start.sh --profile anthropic  Claude      (needs a key, see below)
#   ./start.sh --profile openai     GPT         (needs a key, see below)
#   ./start.sh --chunk-size 300     override chunking for an experiment
#
# Keys go in config/application.yml (gitignored, read automatically) — see
# config/application.yml.example. Prefer that over `export`, which only
# applies to the shell it was typed in.
#
set -euo pipefail

cd "$(dirname "$0")"

if [ -t 1 ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; RED=''; DIM=''; OFF=''
fi
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$1"; }
die()  { printf '  %s✗%s %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }

PROFILE=""
CHUNK_SIZE="${CHUNK_SIZE:-120}"
MIN_CHUNK_CHARS="${MIN_CHUNK_CHARS:-100}"

while [ $# -gt 0 ]; do
  case "$1" in
    --profile)    PROFILE="${2:?--profile needs a value}"; shift 2 ;;
    --chunk-size) CHUNK_SIZE="${2:?--chunk-size needs a value}"; shift 2 ;;
    -h|--help)    sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)            die "Unknown option: $1  (try --help)" ;;
  esac
done

printf '%s\nSpring AI RAG%s\n\n' "$BOLD" "$OFF"

# ── Java 21 (Maven follows JAVA_HOME, not PATH) ───────────────────────────
if [ -z "${JAVA_HOME:-}" ] || ! "${JAVA_HOME}/bin/javac" -version 2>&1 | grep -qE 'javac (2[1-9]|[3-9][0-9])'; then
  for d in /usr/lib/jvm/java-21-* /usr/lib/jvm/*-21-* /Library/Java/JavaVirtualMachines/*21*/Contents/Home; do
    if [ -x "$d/bin/javac" ]; then export JAVA_HOME="$d"; break; fi
  done
fi
[ -n "${JAVA_HOME:-}" ] || die "No JDK 21 found. Run ./setup.sh first."
ok "Java $("${JAVA_HOME}/bin/javac" -version 2>&1 | sed -E 's/javac ([0-9]+).*/\1/') ($JAVA_HOME)"

# ── embedding model must exist, or startup hits the network ───────────────
for f in model.onnx tokenizer.json; do
  [ -f "models/all-MiniLM-L6-v2/$f" ] || die "models/all-MiniLM-L6-v2/$f is missing. Run ./setup.sh first."
done
ok "embedding model present (offline startup)"

# ── PGVector ──────────────────────────────────────────────────────────────
if [ "$(docker inspect -f '{{.State.Health.Status}}' rag-pgvector 2>/dev/null)" != "healthy" ]; then
  printf '  starting PGVector…\n'
  docker compose up -d >/dev/null 2>&1 || die "Could not start PGVector. Is Docker running?"
  for _ in $(seq 1 60); do
    [ "$(docker inspect -f '{{.State.Health.Status}}' rag-pgvector 2>/dev/null)" = "healthy" ] && break
    sleep 1
  done
fi
[ "$(docker inspect -f '{{.State.Health.Status}}' rag-pgvector 2>/dev/null)" = "healthy" ] \
  || die "PGVector unhealthy. Check: docker compose logs pgvector"
ok "PGVector healthy on :5432"

# ── Ollama (only needed for the default provider) ─────────────────────────
if [ -z "$PROFILE" ]; then
  if ! curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    if command -v ollama >/dev/null; then
      printf '  starting ollama serve…\n'
      setsid nohup ollama serve >/tmp/ollama-serve.log 2>&1 </dev/null &
      for _ in $(seq 1 30); do
        curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
        sleep 1
      done
    fi
  fi
  curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1 \
    || die "Ollama is not reachable on :11434. Run ./setup.sh, or use --profile anthropic."
  ok "Ollama up"
else
  ok "provider profile: $PROFILE"
  [ -f config/application.yml ] \
    || warn "config/application.yml not found — the $PROFILE provider needs an API key there."
fi

# ── port check ────────────────────────────────────────────────────────────
if command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ':8080 '; then
  die "Port 8080 is already in use. Stop it with ./stop.sh, or:
      kill \$(ss -ltnp | grep ':8080 ' | grep -oP 'pid=\\K[0-9]+')"
fi

# ── run ───────────────────────────────────────────────────────────────────
ARGS="--rag.chunking.chunk-size=${CHUNK_SIZE} --rag.chunking.min-chunk-size-chars=${MIN_CHUNK_CHARS}"
MVN_OPTS=(-Dspring-boot.run.arguments="$ARGS")
[ -n "$PROFILE" ] && MVN_OPTS+=(-Dspring-boot.run.profiles="$PROFILE")

printf '\n  %sStarting — UI will be at http://localhost:8080%s\n' "$BOLD" "$OFF"
printf '  %sfirst answer takes 30-60s on CPU Ollama; Ctrl+C to stop%s\n\n' "$DIM" "$OFF"

exec ./mvnw spring-boot:run "${MVN_OPTS[@]}"
