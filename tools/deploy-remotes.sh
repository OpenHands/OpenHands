#!/usr/bin/env bash
# deploy-remotes.sh — empurra o agente (código deste repo) da central para os
# hosts remotos usando o próprio agent-server como canal, via HTTPS.
#
# Existe porque os remotos não têm SSH acessível a partir da central: Aqdata
# recusa a porta 22, Lebi/F5 pedem chave. Mas todos já expõem um agent-server
# em HTTPS (é assim que aparecem como backends no Canvas), e ele oferece
# `POST /api/bash/execute_bash_command`. Esse é o canal — atravessa NAT e
# firewall porque reusa a porta que já está aberta.
#
# ESCOPO, POR DESIGN: sincroniza apenas o CÓDIGO do repo (git + build +
# restart). NUNCA toca em configuração local do remoto — settings.json,
# perfis LLM/agente, chaves, config do verboo-rate-limit-proxy. Cada host é
# dono da sua configuração. Para uma correção pontual de infra, use
# `--run-script`, que é uma ação separada e explícita.
#
# Uso:
#   ./tools/deploy-remotes.sh --check                 # só mostra o estado de cada host
#   ./tools/deploy-remotes.sh --dry-run               # mostra o que faria
#   ./tools/deploy-remotes.sh                         # faz o deploy em todos
#   ./tools/deploy-remotes.sh aqdata lebi             # só nesses hosts
#   ./tools/deploy-remotes.sh --run-script tools/fix-openhands-host.sh aqdata
#
# Flags:
#   --check                 só reporta estado (branch, HEAD, serviço), não altera nada
#   --dry-run               mostra os comandos que seriam executados no remoto
#   --branch <b>            branch a sincronizar (default: main)
#   --no-restart            atualiza o código mas não reinicia o serviço
#   --run-script <arquivo>  envia e executa um script local no(s) host(s) e sai
#   --timeout <s>           timeout por comando remoto (default: 600)
#
# CREDENCIAIS — /etc/openhands-deploy/hosts.tsv (chmod 600, fora do repo).
# Uma linha por host, separado por TAB ou '|', comentários com '#':
#
#   # nome    url                                    api_key
#   aqdata    https://aqdata.hands.zadotec.com.br    <chave>
#   lebi      https://lebi.hands.zadotec.com.br      <chave>
#   f5        https://f5.hands.zadotec.com.br        <chave>
#
# A chave de cada host é a mesma que o Canvas usa para falar com aquele
# backend (no navegador: localStorage "openhands-backends").
#
# AVISO DE SEGURANÇA: essa chave dá execução de comando arbitrário como root
# no host. Trate este arquivo como as chaves SSH — 600, nunca no git, e
# rotacione se vazar.

set -euo pipefail

CONFIG="${OPENHANDS_DEPLOY_HOSTS:-/etc/openhands-deploy/hosts.tsv}"
BRANCH="main"
DRY_RUN=0
CHECK_ONLY=0
DO_RESTART=1
RUN_SCRIPT=""
TIMEOUT=600
HOSTS=()

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,48p'; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --no-restart) DO_RESTART=0; shift ;;
    --run-script) RUN_SCRIPT="$2"; shift 2 ;;
    --timeout) TIMEOUT="$2"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "flag desconhecida: $1" >&2; usage ;;
    *) HOSTS+=("$1"); shift ;;
  esac
done

[ -f "$CONFIG" ] || {
  cat >&2 <<EOF
erro: arquivo de hosts não encontrado: $CONFIG

Crie com (chmod 600, fora do repo):

  sudo mkdir -p /etc/openhands-deploy
  sudo install -m 600 /dev/null $CONFIG
  sudo \$EDITOR $CONFIG

Formato (TAB ou '|' como separador):

  aqdata|https://aqdata.hands.zadotec.com.br|<api-key>
  lebi|https://lebi.hands.zadotec.com.br|<api-key>
  f5|https://f5.hands.zadotec.com.br|<api-key>
EOF
  exit 1
}

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$1" "$2" >&2; }
ok()   { printf '\033[1;32m[%s]\033[0m %s\n' "$1" "$2" >&2; }
err()  { printf '\033[1;31m[%s]\033[0m %s\n' "$1" "$2" >&2; }

# Executa um script bash no host remoto. Ecoa stdout/stderr e propaga o exit code.
remote_exec() {
  local name="$1" url="$2" key="$3" script="$4"
  if [ "$DRY_RUN" = "1" ]; then
    echo "--- (dry-run) comando que seria executado em $name ---"
    echo "$script"
    echo "--- fim ---"
    return 0
  fi
  SCRIPT="$script" URL="$url" KEY="$key" TMO="$TIMEOUT" python3 - <<'PY'
import base64, json, os, sys, urllib.request, urllib.error
url = os.environ["URL"].rstrip("/") + "/api/bash/execute_bash_command"
# O endpoint executa via /bin/sh (dash). Mandamos o script em base64 e pedimos
# bash explicitamente: garante bashismos (pipefail) e elimina escaping.
b64 = base64.b64encode(os.environ["SCRIPT"].encode()).decode()
cmd = f"echo {b64} | base64 -d | bash"
body = json.dumps({"command": cmd, "timeout": int(os.environ["TMO"])}).encode()
req = urllib.request.Request(url, data=body, method="POST", headers={
    "Content-Type": "application/json",
    "X-Session-API-Key": os.environ["KEY"],
})
try:
    with urllib.request.urlopen(req, timeout=int(os.environ["TMO"]) + 30) as r:
        d = json.load(r)
except urllib.error.HTTPError as e:
    sys.stderr.write(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:400]}\n")
    sys.exit(2)
except Exception as e:
    sys.stderr.write(f"falha de conexão: {e}\n")
    sys.exit(2)
if d.get("stdout"): sys.stdout.write(d["stdout"])
if d.get("stderr"): sys.stderr.write(d["stderr"])
sys.exit(d.get("exit_code") or 0)
PY
}

# ------------------------------------------------------------- scripts remotos
check_script() {
  cat <<'EOF'
set -uo pipefail
cd /opt/openhands 2>/dev/null || { echo "erro: /opt/openhands não existe"; exit 1; }
echo "host      : $(hostname -s)"
echo "branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
echo "HEAD      : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
dirty=$(git status --porcelain 2>/dev/null | wc -l)
echo "alterações locais não commitadas: $dirty"
for s in openhands agent-canvas; do
  if systemctl cat "$s.service" >/dev/null 2>&1; then
    echo "serviço   : $s ($(systemctl is-active "$s"))"
  fi
done
EOF
}

deploy_script() {
  cat <<EOF
set -euo pipefail
BRANCH="$BRANCH"
DO_RESTART="$DO_RESTART"
EOF
  cat <<'EOF'
cd /opt/openhands

# Nunca descarta trabalho em progresso no remoto.
if [ -n "$(git status --porcelain)" ]; then
  echo "erro: há alterações locais não commitadas em /opt/openhands — aborta." >&2
  git status --short >&2
  exit 1
fi

BEFORE=$(git rev-parse HEAD)
git fetch origin "$BRANCH" --quiet
git checkout "$BRANCH" --quiet
git merge --ff-only "origin/$BRANCH" --quiet
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
  echo "já estava atualizado ($(git rev-parse --short HEAD))"
  CHANGED=""
else
  echo "atualizado: $(git rev-parse --short "$BEFORE") -> $(git rev-parse --short "$AFTER")"
  CHANGED=$(git diff --name-only "$BEFORE" "$AFTER")
  echo "$CHANGED" | sed 's/^/  /'
fi

if echo "$CHANGED" | grep -qE '^package(-lock)?\.json$'; then
  echo "-> package.json mudou: npm ci"
  npm ci
fi
if echo "$CHANGED" | grep -qE '^(src/|app/|public/|vite\.config\.ts|react-router\.config\.ts)'; then
  echo "-> frontend mudou: npm run build"
  npm run build
fi

if [ "$DO_RESTART" = "1" ] && [ -n "$CHANGED" ]; then
  for s in openhands agent-canvas; do
    if systemctl cat "$s.service" >/dev/null 2>&1; then
      echo "-> reiniciando $s"
      systemctl restart "$s"
      sleep 4
      systemctl is-active "$s" >/dev/null && echo "-> $s ativo" || {
        echo "erro: $s não ficou ativo" >&2; exit 1; }
      break
    fi
  done
elif [ -z "$CHANGED" ]; then
  echo "-> nada mudou, sem restart"
else
  echo "-> --no-restart: serviço não reiniciado"
fi
EOF
}

# ------------------------------------------------------------------- main loop
declare -i failures=0 processed=0

while IFS= read -r line; do
  line="${line%%#*}"
  line="$(echo "$line" | sed 's/[[:space:]]*$//')"
  [ -n "$line" ] || continue
  IFS=$'\t|' read -r name url key <<< "$line"
  name="$(echo "$name" | xargs)"; url="$(echo "$url" | xargs)"; key="$(echo "$key" | xargs)"
  [ -n "$name" ] && [ -n "$url" ] && [ -n "$key" ] || { err "config" "linha inválida: $line"; continue; }

  if [ "${#HOSTS[@]}" -gt 0 ]; then
    match=0; for h in "${HOSTS[@]}"; do [ "$h" = "$name" ] && match=1; done
    [ "$match" = "1" ] || continue
  fi

  processed+=1

  if [ -n "$RUN_SCRIPT" ]; then
    [ -f "$RUN_SCRIPT" ] || { err "$name" "script não encontrado: $RUN_SCRIPT"; failures+=1; continue; }
    log "$name" "executando $(basename "$RUN_SCRIPT")..."
    if remote_exec "$name" "$url" "$key" "$(cat "$RUN_SCRIPT")"; then
      ok "$name" "script OK"
    else
      err "$name" "script falhou (exit $?)"; failures+=1
    fi
    continue
  fi

  if [ "$CHECK_ONLY" = "1" ]; then
    log "$name" "consultando estado..."
    remote_exec "$name" "$url" "$key" "$(check_script)" || { err "$name" "falhou"; failures+=1; }
    continue
  fi

  log "$name" "deploy da branch $BRANCH..."
  if remote_exec "$name" "$url" "$key" "$(deploy_script)"; then
    ok "$name" "deploy OK"
  else
    err "$name" "deploy falhou (exit $?)"; failures+=1
  fi
done < "$CONFIG"

[ "$processed" -gt 0 ] || { err "deploy" "nenhum host processado (filtro: ${HOSTS[*]:-nenhum})"; exit 1; }

if [ "$failures" -gt 0 ]; then
  err "deploy" "$failures de $processed host(s) falharam"
  exit 1
fi
ok "deploy" "$processed host(s) processados com sucesso"
