---
card: PROJETOSIN-182
pr: 2
veredicto: PASS
agente: appsec
data: 2026-08-10
re_gate: 2026-08-10
fix_commit: d1ee30c39
ci: review manual services/shared (BE); FE fora deste PR
repo: klebersjunior/OpenHands
branch: feat/fase0-backend-184-185-182
escopo: backend shared auth/capabilities (+ /me/capabilities no Findings)
---

# AppSecurity — PROJETOSIN-182 BE (RBAC shared)

**Veredicto:** PASS

## Re-gate 2026-08-10

Revalidação após `d1ee30c39` (`fix(authz): gate profile header and harden findings ownership`).

### HIGH anterior — Client-controlled profile header — **FECHADO**

- `_profile_header_allowed()` exige `PENTEST_ALLOW_PROFILE_HEADER=1` (somente testes/conftest).
- Precedência runtime: `PENTEST_SESSION_PROFILES` → (header só com flag) → `DEFAULT_PENTEST_PROFILE`.
- Evidência: `test_profile_header_escalation_denied_without_flag` (analyst + header `admin` → caps analyst; sem `pentest.admin.*`).
- Vacuidade: sem a flag, `resolve_profile_for_key(..., profile_header="admin")` retorna default/`pentester`, não `admin`; com flag=1 retorna `admin`.

### Medium anterior — falta regressão de escalation — **FECHADO**

Testes negativos + mapa bate header mesmo com flag (`test_session_profiles_map_beats_header_even_with_flag`).

### Residual (não bloqueante)

- `DEFAULT_PENTEST_PROFILE=pentester` para key sem mapa: adequado a single-tenant local; multi-user deve mapear via `PENTEST_SESSION_PROFILES` / `none`.
- `PENTEST_ALLOW_PROFILE_HEADER` só em conftest — não em compose/Dockerfile.

## Histórico — Gate inicial (FAIL)

**Veredicto na época:** FAIL

O espelho Python de `PROFILE_CAPABILITIES` e `require_capability` estavam alinhados à spec. `GET /api/pentest/me/capabilities` exige auth e retorna 403 sem perfil pentest. Porém o mecanismo de resolução de perfil **confiava no header de cliente** `X-Pentest-Profile`, quebrando a regra de segurança da própria spec 182.

### Finding bloqueante (resolvido no re-gate)

#### HIGH — Client-controlled profile header

```text
Precedência antiga (insegura):
1. PENTEST_SESSION_PROFILES (OK)
2. X-Pentest-Profile (INSEGURO em runtime)
3. DEFAULT_PENTEST_PROFILE (default pentester)
```

Qualquer caller com a session key válida podia enviar `X-Pentest-Profile: admin` e receber `ALL_CAPABILITIES`.

## Relação com FE

UI gating (182 FE) **não** está neste PR; este laudo cobre só BE shared.
