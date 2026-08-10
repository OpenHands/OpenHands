# Architecture Decision Records (ADRs)

ADRs deste projeto usam o template [MADR](https://adr.github.io/madr/).

## Convenção

- Arquivo: `docs/adrs/NNNN-titulo-kebab.md` (número sequencial com 4 dígitos).
- ADRs **aprovadas são imutáveis**. Mudança = nova versão (`NNNN-titulo-v2.md`) proposta via PM.
- Toda feature não-trivial deve referenciar a ADR no PR: `ADR: docs/adrs/0001-….md` + `Plane: PROJETOSIN-<n>`.

## Template mínimo

```markdown
# NNNN. Título

- Status: proposed | accepted | deprecated | superseded by NNNN
- Date: YYYY-MM-DD
- Deciders: PM / Tech Lead / PO

## Context

## Decision

## Consequences

## Alternatives considered
```
