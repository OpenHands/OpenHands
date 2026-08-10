---
card: PROJETOSIN-182
pr: 2
veredicto: FAIL
agente: appsec
data: 2026-08-10
ci: review manual services/shared (BE); FE fora deste PR
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
escopo: backend shared auth/capabilities (+ /me/capabilities no Findings)
---

# AppSecurity — PROJETOSIN-182 BE (RBAC shared)

**Veredicto:** FAIL

## Resumo

O espelho Python de `PROFILE_CAPABILITIES` e `require_capability` estão alinhados à spec. `GET /api/pentest/me/capabilities` exige auth e retorna 403 sem perfil pentest. Porém o mecanismo de resolução de perfil **confia no header de cliente** `X-Pentest-Profile`, quebrando a regra de segurança da própria spec 182.

## Finding bloqueante

### HIGH — Client-controlled profile header

```text
Precedência atual:
1. PENTEST_SESSION_PROFILES (OK)
2. X-Pentest-Profile (INSEGURO em runtime)
3. DEFAULT_PENTEST_PROFILE (default pentester)
```

Qualquer caller com a session key válida pode enviar `X-Pentest-Profile: admin` e receber `ALL_CAPABILITIES`, inclusive rotas admin dos serviços 184/185.

**Remediação mínima para PASS:**

1. Remover o header do caminho de produção **ou** gating por `PENTEST_ALLOW_PROFILE_HEADER=1` (somente testes).
2. Perfis multi-chave **somente** via `PENTEST_SESSION_PROFILES` (server-side).
3. Teste negativo AppSec/QA: key de `analyst`/`client` + header `admin` → continua sem caps admin (403).

## Medium / Low

- Default profile `pentester` para qualquer key sem mapa: amplo para single-tenant local; documentar e preferir `none` + mapa explícito em multi-user.
- Testes shared cobrem só happy-path pentester — falta regressão de escalation.

## Relação com FE

UI gating (182 FE) **não** está neste PR; este laudo cobre só BE shared. Mesmo com FE correta, o bypass server-side permanece explorável.
