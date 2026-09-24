#!/usr/bin/env bash
# fix-openhands-host.sh — aplica em UM host as duas correções dos incidentes de
# 2026-09-11 que deixaram toda conversa nova quebrada. Rode como root no host.
#
#   curl/scp este arquivo para o host e:  sudo bash fix-openhands-host.sh
#
# Correção 1 — proxy de rate-limit duplicando o path
#   /opt/verboo-rate-limit-proxy/config.json tinha upstream_base_url terminando
#   em /router/v1. Como proxy.py usa uma rota catch-all e o agent-server já
#   chama o proxy em .../router/v1/..., a URL final saía duplicada
#   (https://code.verboo.ai/router/v1/router/v1/chat/completions) -> 404, que a
#   UI mostra como `litellm.NotFoundError: OpenAIException - Not Found`.
#   O proxy relê o config a cada requisição: não precisa reiniciar.
#
# Correção 2 — Claude Code (ACP) morre sob root com bypassPermissions
#   O binário embutido em @anthropic-ai/claude-agent-sdk faz:
#     if (bypassPermissions && getuid()===0 && IS_SANDBOX!=="1" && !BUBBLEWRAP)
#       -> console.error(...) ; process.exit(1)
#   Como o serviço roda como root e o provider claude-code usa session mode
#   bypassPermissions, o subprocesso morria na inicialização e o ACP devolvia
#   "Internal error". Começou quando o cache npx foi recriado e trouxe um SDK
#   novo (o claude-agent-acp é pinado, o SDK dele NÃO é).
#   A saída oficial do próprio Claude Code é IS_SANDBOX=1.
#
# Flags:
#   --dry-run     mostra o que faria, sem alterar nada
#   --service X   nome do serviço systemd (default: autodetecta openhands/agent-canvas)

set -euo pipefail

DRY_RUN=0
SERVICE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --service) SERVICE="$2"; shift 2 ;;
    -h|--help) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "flag desconhecida: $1" >&2; exit 1 ;;
  esac
done

[ "$(id -u)" = "0" ] || { echo "erro: rode como root" >&2; exit 1; }

log()  { printf '\033[1;34m[%s]\033[0m %s\n' "$(hostname -s)" "$*"; }
warn() { printf '\033[1;33m[%s] aviso:\033[0m %s\n' "$(hostname -s)" "$*"; }
run()  { if [ "$DRY_RUN" = "1" ]; then echo "  (dry-run) $*"; else eval "$@"; fi; }

changed=0

# ---------------------------------------------------------------- correção 1
PROXY_CFG=/opt/verboo-rate-limit-proxy/config.json
if [ -f "$PROXY_CFG" ]; then
  current=$(python3 -c "import json;print(json.load(open('$PROXY_CFG'))['upstream_base_url'])")
  log "proxy: upstream_base_url atual = $current"
  if [[ "$current" == */router/v1 || "$current" == */router/v1/ ]]; then
    fixed="${current%/}"; fixed="${fixed%/router/v1}"
    log "proxy: CORRIGINDO -> $fixed"
    run "cp -a '$PROXY_CFG' '$PROXY_CFG.bak.\$(date +%s)'"
    run "python3 - <<EOF
import json
p='$PROXY_CFG'
d=json.load(open(p))
d['upstream_base_url']='$fixed'
json.dump(d, open(p,'w'), indent=2)
open(p,'a').write('\n')
EOF"
    changed=1
  else
    log "proxy: já correto, nada a fazer"
  fi
else
  log "proxy: $PROXY_CFG não existe neste host — pulando correção 1"
fi

# ---------------------------------------------------------------- correção 2
if [ -z "$SERVICE" ]; then
  for s in openhands agent-canvas; do
    if systemctl list-unit-files "$s.service" >/dev/null 2>&1 && \
       systemctl cat "$s.service" >/dev/null 2>&1; then SERVICE="$s"; break; fi
  done
fi

if [ -z "$SERVICE" ]; then
  warn "nenhum serviço systemd openhands/agent-canvas encontrado — pulando correção 2"
else
  log "serviço detectado: $SERVICE.service"
  svc_user=$(systemctl show -p User --value "$SERVICE.service" 2>/dev/null)
  svc_user=${svc_user:-root}
  if [ "$svc_user" != "root" ]; then
    log "serviço roda como '$svc_user' (não-root) — correção 2 desnecessária"
  elif systemctl show -p Environment --value "$SERVICE.service" 2>/dev/null | grep -q "IS_SANDBOX=1"; then
    log "IS_SANDBOX=1 já configurado, nada a fazer"
  else
    OVR="/etc/systemd/system/$SERVICE.service.d/override.conf"
    log "CORRIGINDO: criando $OVR com IS_SANDBOX=1"
    run "mkdir -p '$(dirname "$OVR")'"
    run "cat > '$OVR' <<'EOF'
[Service]
# O provider ACP claude-code usa session mode bypassPermissions, que o Claude
# Code recusa sob uid 0 a menos que IS_SANDBOX=1 esteja definido:
#   \"--dangerously-skip-permissions cannot be used with root/sudo privileges\"
# Sem isso o subprocesso morre com exit(1) e o ACP devolve \"Internal error\".
Environment=IS_SANDBOX=1
EOF"
    run "systemctl daemon-reload"
    log "reiniciando $SERVICE (derruba conexões ativas por alguns segundos)"
    run "systemctl restart '$SERVICE'"
    changed=1
  fi
fi

# ---------------------------------------------------------------- validação
if [ "$DRY_RUN" = "1" ]; then
  log "dry-run concluído — nada foi alterado"
  exit 0
fi

sleep 5
log "--- validação ---"
[ -n "$SERVICE" ] && log "serviço: $(systemctl is-active "$SERVICE")"
if [ -n "$SERVICE" ]; then
  pid=$(systemctl show -p MainPID --value "$SERVICE")
  if [ -n "$pid" ] && [ "$pid" != "0" ]; then
    if tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q '^IS_SANDBOX=1'; then
      log "IS_SANDBOX=1 presente no processo"
    else
      warn "IS_SANDBOX=1 NÃO chegou no processo"
    fi
  fi
fi
if [ -f "$PROXY_CFG" ]; then
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
    -X POST http://127.0.0.1:18010/router/v1/chat/completions \
    -H 'Content-Type: application/json' -d '{}' || echo "000")
  case "$code" in
    401|403) log "proxy OK: rota chega ao upstream (HTTP $code = falta auth, esperado sem chave)" ;;
    404)     warn "proxy ainda responde 404 — path pode continuar duplicado" ;;
    *)       warn "proxy respondeu HTTP $code (inesperado, verifique manualmente)" ;;
  esac
fi
[ "$changed" = "1" ] && log "correções aplicadas" || log "nada precisou ser alterado"
log "teste enviando uma mensagem numa conversa nova; se falhar, veja:"
log "  journalctl -u ${SERVICE:-openhands} -n 100   e   ~/.openhands/agent-canvas/logs/"
