# CLAUDE.md

Instruções para agentes de IA que trabalham neste repositório.

O modelo completo de orquestração, o time de agentes, os gates de bloqueio e as convenções do projeto estão em **[AGENTS.md](AGENTS.md)** — leia antes de agir. Este arquivo destaca as regras que não admitem exceção.

## Paralelismo de agentes — decisão do PO (2026-08-01)

**Agentes PODEM atuar em paralelo** quando o trabalho for claramente independente (cards/branches/worktrees distintos, ou gates em PRs diferentes).

A regra de 2026-07-30 (“um agente por vez, sem exceção”) foi **substituída** em 2026-08-01. Mantêm-se salvaguardas:

- **Um escritor por checkout** — nunca dois agentes escrevendo na mesma working tree; preferir worktree/branch separada.
- **Gates no mesmo PR** — sequenciais se ambos mutam laudo/fixture/testes no mesmo tree.
- **Antes de redespachar por “morte”**, confirme (`git log`, PR, comentário no card). **Lentidão ≠ morte.** Silêncio não encerra thread.

**Quando duas linhas já colidiram:** designe **uma** dona da branch, não acorde a outra, e trie a árvore hunk por hunk. `reset --hard` / `checkout --` / `clean` destroem correção não commitada. Cuidado com `git checkout <ref> -- <arquivo>`: deixa a versão antiga no **índice**, e o `checkout --` seguinte restaura dela — a mutação sobrevive com `git diff --stat` vazio.

Detalhes: [AGENTS.md § Paralelismo de agentes](AGENTS.md).

## Estado observado envelhece

Antes de afirmar estado de repositório — num gate, num card ou num handoff — rode `git fetch` e reconfirme. Verificar custa uma chamada de comando.

## Gate não se auto-assina

Quem escreveu o código não emite o laudo de QA nem de AppSec. Gate verifica com cenário próprio; asserções positivas devem poder ser tornadas vácuas se o controle não existir.

## Verificação antes de gate / merge

```bash
npm run lint
npm test
npm run build
```

Quando o escopo tocar stack/E2E: `npm run test:e2e:mock-llm` (ou o subset mapeado em `tests/e2e/mock-llm/test-mapping.json`). Laudos em `docs/gates/<card>/`. Detalhe: [AGENTS.md](AGENTS.md), [docs/gates/README.md](docs/gates/README.md).

## O card é o estado da verdade

Todo trabalho tem work item no Plane (`PROJETOSIN-<n>`), Module **OpenHands**, atualizado **durante** e não no fim. Trabalho entregue com card desatualizado não conta como entregue. Ver [AGENTS.md § Rastreamento no Plane](AGENTS.md).

## Onde está o resto

| Assunto | Arquivo |
|---|---|
| Time de agentes, fluxo, gates, convenções | [AGENTS.md](AGENTS.md) |
| Notas técnicas do codebase | [AGENTS.md § Repository Notes](AGENTS.md#repository-notes) |
| Definições de cada agente | [`.claude/agents/`](.claude/agents/) |
| ADRs (imutáveis após aprovação) | [`docs/adrs/`](docs/adrs/) |
| Laudos de gate por card | [`docs/gates/`](docs/gates/) |
| Arquitetura | [`docs/architecture.md`](docs/architecture.md) |
