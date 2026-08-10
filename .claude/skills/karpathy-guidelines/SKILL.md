---
name: karpathy-guidelines
description: Diretrizes de codificação estilo Karpathy — simplicidade, mudanças cirúrgicas, verificação. Use ao escrever, revisar ou refatorar código.
---

# Karpathy guidelines

Ao escrever, revisar ou refatorar código:

1. **Pensar antes de codar** — explicite premissas; se houver múltiplas interpretações, apresente-as; se algo está obscuro, pare e pergunte.
2. **Simplicidade primeiro** — o mínimo que resolve; nada especulativo, nenhuma abstração de uso único, nenhum tratamento de erro para cenário impossível.
3. **Mudanças cirúrgicas** — toque só no necessário; não "melhore" código adjacente; case o estilo existente; remova só os órfãos que suas mudanças criaram; código morto pré-existente se menciona, não se apaga.
4. **Execução orientada a meta** — transforme a tarefa em critério verificável (teste que reproduz o bug; testes que passam antes e depois) e itere até verificar.

Trade-off: cautela sobre velocidade; para tarefas triviais, use bom senso.
