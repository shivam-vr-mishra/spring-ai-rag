#!/usr/bin/env bash
#
# One-time setup for the Spring AI RAG POC.
#
#   ./setup.sh
#
# Checks prerequisites, fetches the local embedding model, pulls the Ollama
# chat model, and resolves Maven dependencies. Safe to re-run: every step is
# skipped if already done.
#
set -euo pipefail

cd "$(dirname "$0")"

# ── output helpers ────────────────────────────────────────────────────────
if [ -t 1 ]; then
  BOLD=$'\033[1m'; GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'; DIM=$'\033[2m'; OFF=$'\033[0m'
else
  BOLD=''; GREEN=''; YELLOW=''; RED=''; DIM=''; OFF=''
fi
step() { printf '\n%s▸ %s%s\n' "$BOLD" "$1" "$OFF"; }
ok()   { printf '  %s✓%s %s\n' "$GREEN" "$OFF" "$1"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$1"; }
die()  { printf '  %s✗%s %s\n' "$RED" "$OFF" "$1" >&2; exit 1; }

MODEL_DIR="models/all-MiniLM-L6-v2"
ONNX_SHA="e3dde332c13808c718680e7bf74a574e7e5d06f55bd6e1527e51509dcb8206f3"
ONNX_URL="https://media.githubusercontent.com/media/spring-projects/spring-ai/main/models/spring-ai-transformers/src/main/resources/onnx/all-MiniLM-L6-v2/model.onnx"
SPRING_AI_VERSION="2.0.2"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.2:3b}"

printf '%s\nSpring AI RAG — setup%s\n' "$BOLD" "$OFF"

# ── 1. Java 21+ ───────────────────────────────────────────────────────────
step "Java"
if [ -n "${JAVA_HOME:-}" ] && [ -x "${JAVA_HOME}/bin/javac" ]; then
  JAVAC="${JAVA_HOME}/bin/javac"
else
  JAVAC="$(command -v javac || true)"
fi
[ -n "$JAVAC" ] || die "javac not found. Install JDK 21+ (e.g. sudo apt install openjdk-21-jdk)."

JAVA_MAJOR="$("$JAVAC" -version 2>&1 | sed -E 's/javac ([0-9]+).*/\1/')"
if [ "$JAVA_MAJOR" -lt 21 ]; then
  # Maven follows JAVA_HOME, not PATH -- a mismatch here produces the
  # confusing "release version 21 not supported" rather than a clear error.
  FOUND=""
  for d in /usr/lib/jvm/java-21-* /usr/lib/jvm/*-21-* /Library/Java/JavaVirtualMachines/*21*/Contents/Home; do
    [ -x "$d/bin/javac" ] && { FOUND="$d"; break; }
  done
  [ -n "$FOUND" ] || die "Java $JAVA_MAJOR found, but 21+ is required. Install JDK 21."
  warn "JAVA_HOME points at Java $JAVA_MAJOR; using $FOUND instead."
  warn "Make it permanent:  echo 'export JAVA_HOME=$FOUND' >> ~/.bashrc"
  export JAVA_HOME="$FOUND"
else
  ok "Java $JAVA_MAJOR"
fi

# ── 2. Docker ─────────────────────────────────────────────────────────────
step "Docker"
command -v docker >/dev/null || die "docker not found. Install Docker: https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || die "Docker is installed but not running. Start the Docker daemon."
docker compose version >/dev/null 2>&1 || die "'docker compose' not available. Install the Compose plugin."
ok "Docker $(docker --version | sed -E 's/Docker version ([^,]+),.*/\1/') with Compose"

# ── 3. Ollama (default chat provider — no API key) ────────────────────────
step "Ollama"
if ! command -v ollama >/dev/null; then
  warn "ollama not found."
  printf '    Install it with:  %scurl -fsSL https://ollama.com/install.sh | sh%s\n' "$DIM" "$OFF"
  printf '    Or skip Ollama and use a hosted provider:\n'
  printf '      cp config/application.yml.example config/application.yml   # add your key\n'
  printf '      ./start.sh --profile anthropic\n'
else
  ok "ollama $(ollama --version 2>/dev/null | head -1 | sed -E 's/.*version is //')"

  if ! curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    printf '  starting ollama serve in the background…\n'
    setsid nohup ollama serve >/tmp/ollama-serve.log 2>&1 </dev/null &
    for _ in $(seq 1 30); do
      curl -fsS -m 2 http://localhost:11434/api/tags >/dev/null 2>&1 && break
      sleep 1
    done
  fi

  if curl -fsS -m 3 http://localhost:11434/api/tags >/dev/null 2>&1; then
    if curl -fsS -m 5 http://localhost:11434/api/tags | grep -q "\"$OLLAMA_MODEL\""; then
      ok "model $OLLAMA_MODEL already pulled"
    else
      printf '  pulling %s (~2GB, one time)…\n' "$OLLAMA_MODEL"
      ollama pull "$OLLAMA_MODEL"
      ok "model $OLLAMA_MODEL ready"
    fi
  else
    warn "could not reach ollama on :11434 — start it manually with 'ollama serve'"
  fi
fi

# ── 4. Maven dependencies ─────────────────────────────────────────────────
step "Maven dependencies"
./mvnw -q -B dependency:resolve \
  || die "Dependency resolution failed. If you see AccessDeniedException under ~/.m2, run:
      sudo chown -R \"\$USER:\$USER\" ~/.m2/repository"
ok "dependencies resolved"

# ── 5. Local embedding model ──────────────────────────────────────────────
# Both files are fetched from GitHub on every startup at Spring AI's defaults,
# into a randomly named temp dir. Pinning them locally makes startup offline
# and reproducible -- see docs/LAYER-1-INGESTION.md.
step "Embedding model (all-MiniLM-L6-v2, 384d)"
mkdir -p "$MODEL_DIR"

JAR="$HOME/.m2/repository/org/springframework/ai/spring-ai-transformers/${SPRING_AI_VERSION}/spring-ai-transformers-${SPRING_AI_VERSION}.jar"
if [ -f "$MODEL_DIR/tokenizer.json" ]; then
  ok "tokenizer.json present"
elif [ -f "$JAR" ]; then
  # The jar's tokenizer.json is real (unlike model.onnx) -- no download needed.
  unzip -o -j -q "$JAR" onnx/all-MiniLM-L6-v2/tokenizer.json -d "$MODEL_DIR"
  ok "tokenizer.json extracted from the Spring AI jar"
else
  die "Could not find $JAR — Maven resolution may have failed."
fi

verify_onnx() {
  [ -f "$MODEL_DIR/model.onnx" ] || return 1
  [ "$(sha256sum "$MODEL_DIR/model.onnx" | cut -d' ' -f1)" = "$ONNX_SHA" ]
}

if verify_onnx; then
  ok "model.onnx present and checksum verified"
else
  [ -f "$MODEL_DIR/model.onnx" ] && warn "model.onnx checksum mismatch — re-downloading"
  # The copy inside the jar is a 133-byte Git LFS pointer, not the model.
  printf '  downloading model.onnx (86MB, one time)…\n'
  curl -fL --progress-bar -o "$MODEL_DIR/model.onnx" "$ONNX_URL" \
    || die "Download failed. Check your network and re-run ./setup.sh"
  verify_onnx || die "Checksum mismatch after download. Expected $ONNX_SHA"
  ok "model.onnx downloaded and verified"
fi

# ── 6. Database ───────────────────────────────────────────────────────────
step "PGVector"
docker compose up -d >/dev/null 2>&1
for _ in $(seq 1 60); do
  [ "$(docker inspect -f '{{.State.Health.Status}}' rag-pgvector 2>/dev/null)" = "healthy" ] && break
  sleep 1
done
[ "$(docker inspect -f '{{.State.Health.Status}}' rag-pgvector 2>/dev/null)" = "healthy" ] \
  || die "PGVector did not become healthy. Check: docker compose logs pgvector"
ok "PGVector healthy on :5432"

# ── done ──────────────────────────────────────────────────────────────────
printf '\n%s✓ Setup complete.%s\n\n' "$GREEN$BOLD" "$OFF"
printf '  Start the app:   %s./start.sh%s\n' "$BOLD" "$OFF"
printf '  Then open:       %shttp://localhost:8080%s\n\n' "$BOLD" "$OFF"
printf '  %sThe default provider is Ollama — local and free, no API key.%s\n' "$DIM" "$OFF"
printf '  %sFor Claude or OpenAI, see config/application.yml.example.%s\n\n' "$DIM" "$OFF"
