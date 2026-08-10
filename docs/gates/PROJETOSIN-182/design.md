---
card: PROJETOSIN-182
pr: 3
veredicto: PASS
agente: design
data: 2026-08-10
repo: klebersjunior/OpenHands
branch: feat/fase0-frontend-182-183
---

# Design Review — PROJETOSIN-182 CapabilityGate (UI)

**Veredicto:** PASS

O card 182 é majoritariamente contrato de capabilities + middleware. A superfície de UI própria é o wrapper `CapabilityGate`; o visual pentest vive nos consumidores (card 183).

## Escopo revisado

| Superfície | Arquivo |
|---|---|
| `CapabilityGate` | `src/components/features/pentest/capability-gate.tsx` |
| Consumo no seletor / autonomia | `workspace-type-selector.tsx`, `pentest-workspace-fields.tsx` |

Spec: `docs/specs/fase-0/182-rbac-feature-gating.md` (sem strings de UI obrigatórias neste card).

## Achados

- Gate renderiza `children` ou `fallback` (default `null`) — padrão correto de **feature hide**, sem empty shell confuso.
- Consumidores 183 usam hide (não disabled+tooltip) para opção Pentest e chip Autonomous — consistente com AC-182-1/2 e AC-183-1.
- Sem magic strings; componente sem chrome visual próprio — nada a validar em tokens/HeroUI além do contrato de composição.

## Issues

Nenhuma issue de UI/a11y bloqueante no wrapper.

## Veredicto

**PASS** — CapabilityGate adequado como primitiva de gating visual. Detalhe de polish dos consumidores: ver `docs/gates/PROJETOSIN-183/design.md`.
