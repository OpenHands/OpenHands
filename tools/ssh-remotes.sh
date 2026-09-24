#!/usr/bin/env bash
# ssh-remotes.sh — opera os hosts remotos do OpenHands por SSH.
#
# Irmão do deploy-remotes.sh: mesma interface, transporte diferente. Use este
# quando você TEM acesso SSH aos remotos (ex.: rodando do PC do João, cuja
# origem é liberada no firewall); use o deploy-remotes.sh quando só houver
# HTTPS (ex.: rodando da central, que não alcança a porta 22).
#
# Requer bash — no Windows, rode pelo Git Bash ou WSL, não pelo PowerShell.
#
# Uso:
#   ./tools/ssh-remotes.sh --check                 # conectividade + estado de cada host
#   ./tools/ssh-remotes.sh --dry-run               # mostra o que faria
#   ./tools/ssh-remotes.sh                         # deploy do código em todos
#   ./tools/ssh-remotes.sh aqdata                  # só nesse host
#   ./tools/ssh-remotes.sh --run-script tools/fix-openhands-host.sh
#
# Flags:
#   --check                 só diagnostica (porta 22, login, git, serviço)
#   --dry-run               não altera nada
#   --branch <b>            branch a sincronizar (default: main)
#   --no-restart            atualiza sem reiniciar o serviço
#   --run-script <arquivo>  envia e executa um script local no(s) host(s)
#   --user <u>              usuário SSH (default: root, ou o do hosts.tsv)
#
# HOSTS — ./tools/remotes.tsv ou $OPENHANDS_SSH_HOSTS.
# Uma linha por host: nome|destino-ssh[:porta]
#
#   aqdata|root@aqdata.hands.exemplo.com.br:22
#   lebi|root@lebi.hands.exemplo.com.br
#   f5|root@f5.hands.exemplo.com.br
#
# Este arquivo NÃO contém segredo (só hostnames) — a autenticação é por chave
# SSH do seu agente. Ainda assim, se ele mapear sua infra, mantenha fora do
# repo público: aponte $OPENHANDS_SSH_HOSTS para um caminho privado.

set -uo pipefail

CONFIG="${OPENHANDS_SSH_HOSTS:-$(dirname "$0")/remotes.tsv}"
BRANCH="main"
DRY_RUN=0
CHECK_ONLY=0
DO_RESTART=1
RUN_SCRIPT=""
SSH_USER=""
HOSTS=()

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//' | sed -n '2,40p'; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --check) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --branch) BRANCH="$2"; shift 2 ;;
    --no-restart) DO_RESTART=0; shift ;;
    --run-script) RUN_SCRIPT="$2"; shift 2 ;;
    --user) SSH_USER="$2"; shift 2 ;;
    -h|--help) usage ;;
    -*) echo "flag desconhecida: $1" >&2; usage ;;
    *) HOSTS+=("$1"); shift ;;
  esac
done

[ -f "$CONFIG" ] || {
  cat >&2 <<EOF
erro: lista de hosts não encontrada: $CONFIG

Crie com uma linha por host (nome|destino-ssh[:porta]):

  printf '%s\n' \\
    'aqdata|root@aqdata.hands.exemplo.com.br' \\
    'lebi|root@lebi.hands.exemplo.com.br' \\
    'f5|root@f5.hands.exemplo.com.br' > $CONFIG
EOF
  exit 1
}

log() { printf '\033[1;34m[%s]\033[0m %s\n' "$1" "$2" >&2; }
ok()  { printf '\033[1;32m[%s]\033[0m %s\n' "$1" "$2" >&2; }
err() { printf '\033[1;31m[%s]\033[0m %s\n' "$1" "$2" >&2; }

# ---------------------------------------------------------------- diagnóstico
diagnose() {
  local name="$1" dest="$2" port="$3"
  local hostonly="${dest#*@}"

  if command -v nc >/dev/null 2>&1; then
    if ! nc -z -w5 "$hostonly" "$port" 2>/dev/null; then
      err "$name" "porta $port fechada/inacessível em $hostonly"
      err "$name" "  causas comuns: SSH em outra porta, firewall liberando só certos IPs,"
      err "$name" "  ou sshd desligado. Confira no painel do provedor ou no console."
      return 1
    fi
  fi

  local out
  out=$(ssh "${SSH_OPTS[@]}" -p "$port" "$dest" 'echo ok' 2>&1)
  if [ "$out" != "ok" ]; then
    err "$name" "SSH recusou: $(echo "$out" | tail -1)"
    case "$out" in
      *"Permission denied"*)
        err "$name" "  instale sua chave pública com:"
        err "$name" "    ssh-copy-id -p $port $dest"
        err "$name" "  (precisa de senha ou acesso pelo console na primeira vez)" ;;
    esac
    return 1
  fi
  return 0
}

# ------------------------------------------------------------ scripts remotos
check_script() {
  cat <<'EOF'
set -uo pipefail
echo "host      : $(hostname -s)"
echo "usuário   : $(id -un)"
if cd /opt/openhands 2>/dev/null; then
  echo "branch    : $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
  echo "HEAD      : $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
  echo "não commitado: $(git status --porcelain 2>/dev/null | wc -l) arquivo(s)"
else
  echo "aviso     : /opt/openhands não existe"
fi
for s in openhands agent-canvas; do
  systemctl cat "$s.service" >/dev/null 2>&1 && echo "serviço   : $s ($(systemctl is-active "$s"))"
done
EOF
}

deploy_script() {
  printf 'BRANCH=%q\nDO_RESTART=%q\n' "$BRANCH" "$DO_RESTART"
  cat <<'EOF'
set -euo pipefail
cd /opt/openhands

if [ -n "$(git status --porcelain)" ]; then
  echo "erro: alterações locais não commitadas em /opt/openhands — aborta." >&2
  git status --short >&2
  exit 1
fi

BEFORE=$(git rev-parse HEAD)
git fetch origin "$BRANCH" --quiet
git checkout "$BRANCH" --quiet
git merge --ff-only "origin/$BRANCH" --quiet
AFTER=$(git rev-parse HEAD)

if [ "$BEFORE" = "$AFTER" ]; then
  echo "já atualizado ($(git rev-parse --short HEAD))"; CHANGED=""
else
  echo "atualizado: $(git rev-parse --short "$BEFORE") -> $(git rev-parse --short "$AFTER")"
  CHANGED=$(git diff --name-only "$BEFORE" "$AFTER"); echo "$CHANGED" | sed 's/^/  /'
fi

echo "$CHANGED" | grep -qE '^package(-lock)?\.json$' && { echo "-> npm ci"; npm ci; }
echo "$CHANGED" | grep -qE '^(src/|app/|public/|vite\.config\.ts|react-router\.config\.ts)' && { echo "-> npm run build"; npm run build; }

if [ "$DO_RESTART" = "1" ] && [ -n "$CHANGED" ]; then
  for s in openhands agent-canvas; do
    if systemctl cat "$s.service" >/dev/null 2>&1; then
      echo "-> reiniciando $s"; systemctl restart "$s"; sleep 4
      systemctl is-active "$s" >/dev/null && echo "-> $s ativo" || { echo "erro: $s não subiu" >&2; exit 1; }
      break
    fi
  done
fi
EOF
}

# Envia o script por stdin e executa com bash (evita depender do shell de login).
remote_run() {
  local dest="$1" port="$2" script="$3"
  if [ "$DRY_RUN" = "1" ]; then
    echo "--- (dry-run) seria executado em $dest ---"; echo "$script"; echo "--- fim ---"; return 0
  fi
  printf '%s' "$script" | ssh "${SSH_OPTS[@]}" -p "$port" "$dest" 'bash -s'
}

# ------------------------------------------------------------------ main loop
declare -i failures=0 processed=0

while IFS= read -r line; do
  line="${line%%#*}"; line="$(echo "$line" | sed 's/[[:space:]]*$//')"
  [ -n "$line" ] || continue
  IFS='|' read -r name dest <<< "$line"
  name="$(echo "$name" | xargs)"; dest="$(echo "$dest" | xargs)"
  [ -n "$name" ] && [ -n "$dest" ] || { err "config" "linha inválida: $line"; continue; }

  port=22
  case "$dest" in *:[0-9]*) port="${dest##*:}"; dest="${dest%:*}" ;; esac
  [ -n "$SSH_USER" ] && dest="${SSH_USER}@${dest#*@}"

  if [ "${#HOSTS[@]}" -gt 0 ]; then
    match=0; for h in "${HOSTS[@]}"; do [ "$h" = "$name" ] && match=1; done
    [ "$match" = "1" ] || continue
  fi
  processed+=1

  if [ "$DRY_RUN" = "0" ] && ! diagnose "$name" "$dest" "$port"; then failures+=1; continue; fi

  if [ -n "$RUN_SCRIPT" ]; then
    [ -f "$RUN_SCRIPT" ] || { err "$name" "script não encontrado: $RUN_SCRIPT"; failures+=1; continue; }
    log "$name" "executando $(basename "$RUN_SCRIPT")..."
    remote_run "$dest" "$port" "$(cat "$RUN_SCRIPT")" && ok "$name" "OK" || { err "$name" "falhou"; failures+=1; }
  elif [ "$CHECK_ONLY" = "1" ]; then
    log "$name" "estado:"
    remote_run "$dest" "$port" "$(check_script)" || { err "$name" "falhou"; failures+=1; }
  else
    log "$name" "deploy da branch $BRANCH..."
    remote_run "$dest" "$port" "$(deploy_script)" && ok "$name" "deploy OK" || { err "$name" "deploy falhou"; failures+=1; }
  fi
done < "$CONFIG"

[ "$processed" -gt 0 ] || { err "ssh-remotes" "nenhum host processado"; exit 1; }
[ "$failures" -eq 0 ] || { err "ssh-remotes" "$failures de $processed host(s) falharam"; exit 1; }
ok "ssh-remotes" "$processed host(s) OK"
