#!/usr/bin/env bash
# refresh-remotes.sh — atualiza um checkout git deste repo em uma ou mais
# máquinas remotas (via SSH) e reinicia o agent-canvas, para que elas peguem
# mudanças como o novo tools/codegraph_tool.py.
#
# Pressupõe que cada remoto já roda o agent-canvas a partir de um checkout
# git deste repositório (não via npx/registry, não via Docker) — ex.:
#   node bin/agent-canvas.mjs --public
# gerenciado por systemd, tmux, ou outro supervisor.
#
# Uso:
#   ./tools/refresh-remotes.sh <host1> [host2 ...]
#   ./tools/refresh-remotes.sh --user deploy --path /opt/openhands --service agent-canvas host1 host2
#
# Variáveis de ambiente / flags:
#   --user <ssh-user>       usuário SSH (default: usuário atual)
#   --path <repo-path>      caminho do checkout no remoto (default: /opt/openhands)
#   --service <name>        nome do serviço systemd a reiniciar (default: agent-canvas)
#   --no-restart            só atualiza o código, não reinicia o serviço
#   --branch <branch>       branch a atualizar (default: main)
#
# O script é idempotente e para no primeiro erro (set -e) — se um host falhar,
# os demais já processados continuam rodando a versão nova; conserte o host
# problemático e rode de novo só para ele.

set -euo pipefail

SSH_USER="${SSH_USER:-}"
REPO_PATH="/opt/openhands"
SERVICE_NAME="agent-canvas"
BRANCH="main"
DO_RESTART=1
HOSTS=()

usage() {
  grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,26p'
  exit 1
}

while [ $# -gt 0 ]; do
  case "$1" in
    --user) SSH_USER="$2"; shift 2 ;;
    --path) REPO_PATH="$2"; shift 2 ;;
    --service) SERVICE_NAME="$2"; shift 2 ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --no-restart) DO_RESTART=0; shift ;;
    -h|--help) usage ;;
    *) HOSTS+=("$1"); shift ;;
  esac
done

[ "${#HOSTS[@]}" -gt 0 ] || { echo "erro: informe pelo menos um host" >&2; usage; }

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$1" "$2" >&2; }

refresh_host() {
  local host="$1"
  local target="$host"
  [ -n "$SSH_USER" ] && target="${SSH_USER}@${host}"

  log "$host" "conectando..."

  # shellcheck disable=SC2087
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$target" bash -s -- \
    "$REPO_PATH" "$BRANCH" "$SERVICE_NAME" "$DO_RESTART" <<'REMOTE_SCRIPT'
set -euo pipefail
REPO_PATH="$1"; BRANCH="$2"; SERVICE_NAME="$3"; DO_RESTART="$4"

cd "$REPO_PATH"

echo "-> git status antes do pull:"
git status --short --branch

# Nunca descarta trabalho em progresso silenciosamente.
if [ -n "$(git status --porcelain)" ]; then
  echo "erro: há alterações locais não commitadas em $REPO_PATH — aborta." >&2
  exit 1
fi

BEFORE=$(git rev-parse HEAD)
git fetch origin "$BRANCH"
git checkout "$BRANCH"
git merge --ff-only "origin/$BRANCH"
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
  echo "-> já estava atualizado ($AFTER)"
else
  echo "-> atualizado: $BEFORE -> $AFTER"
  echo "-> arquivos alterados:"
  git diff --name-only "$BEFORE" "$AFTER"
fi

# Só reinstala/rebuilda se algo relevante mudou (poupa tempo em refreshes
# que só tocam tools/*.py, como o codegraph_tool).
if git diff --name-only "$BEFORE" "$AFTER" | grep -qE '^(package(-lock)?\.json)$'; then
  echo "-> package.json mudou, rodando npm ci"
  npm ci
fi
if git diff --name-only "$BEFORE" "$AFTER" | grep -qE '^(app/|src/|public/|react-router\.config\.ts|vite\.config\.ts)'; then
  echo "-> frontend mudou, rodando npm run build"
  npm run build
fi

if [ "$DO_RESTART" = "1" ]; then
  if systemctl is-enabled "$SERVICE_NAME" >/dev/null 2>&1; then
    echo "-> reiniciando serviço systemd '$SERVICE_NAME'"
    sudo systemctl restart "$SERVICE_NAME"
    sleep 2
    systemctl is-active "$SERVICE_NAME" && echo "-> serviço ativo" || {
      echo "erro: serviço não ficou ativo, veja: journalctl -u $SERVICE_NAME -n 100" >&2
      exit 1
    }
  elif tmux has-session -t canvas 2>/dev/null; then
    echo "-> matando sessão tmux 'canvas' (reinicie manualmente com o comando de sempre)"
    tmux kill-session -t canvas
    echo "aviso: sessão tmux morta, suba de novo com:"
    echo "  tmux new-session -d -s canvas 'node bin/agent-canvas.mjs --public'"
  else
    echo "aviso: nenhum serviço systemd '$SERVICE_NAME' nem sessão tmux 'canvas' encontrados — reinicie manualmente." >&2
  fi
fi
REMOTE_SCRIPT

  log "$host" "OK"
}

for h in "${HOSTS[@]}"; do
  refresh_host "$h"
done

log "done" "todos os hosts processados: ${HOSTS[*]}"
