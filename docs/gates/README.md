# Gates de qualidade

Laudos emitidos pelos agentes de gate (Design, QA, AppSecurity) para cada card Plane.

## Layout

```
docs/gates/
  PROJETOSIN-<n>/
    design.md      # gate de UI (quando houver UI)
    qa.md          # gate de AC / regressão
    appsec.md      # gate de segurança
```

## Regras

1. **Revisor ≠ autor.** Quem implementou não assina o laudo do próprio PR.
2. Laudo em `docs/gates/` **e** review formal no PR GitHub.
3. Frontmatter sugerido:

```yaml
---
card: PROJETOSIN-12
pr: 123
veredicto: PASS   # ou FAIL
agente: qa        # design | qa | appsec
data: 2026-08-10
ci: npm-test+lint+build
---
```

4. **FAIL** → label `Blocked` no card + comentário com remediação.
5. Merge só com todos os gates aplicáveis em PASS (Tech Lead).

Ver [AGENTS.md](../AGENTS.md) § Gates de bloqueio.
