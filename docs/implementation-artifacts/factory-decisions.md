# Décisions factory — 💄 Nest sub-agent conversations in menu (688e7f29)

> Convention BBY : toute question non bloquante est consignée ici avec
> l'hypothèse retenue, puis le travail continue. Aucune question bloquante
> identifiée sur ce ticket.

## D1 — Où livrer le correctif (2026-10-10)

**Question** : aucune cible de MR n'existait (aucun fork OpenHands chez
Bilail/bby-dev, le dépôt `/opt/agent-canvas` local est un build déployé sans
git). Où pousser la branche `agent/<ticket>` ?

**Hypothèse retenue** : MR brouillon **vers le dépôt upstream**
`OpenHands/OpenHands` depuis un fork à créer sous l'org `bby-dev` (le token
GitHub est le compte Bilail, admin de l'org). Raisons :

1. La fonctionnalité manquante vit dans le frontend upstream
   (`conversation-panel.tsx`, `agent-server-adapter.ts`) — le bug est
   reproductible sur la main upstream, pas seulement sur l'install locale.
2. Le patch du bundle minifié de `/opt/agent-canvas` serait écrasé à la
   prochaine mise à jour de l'image : non maintenable.
3. La règle factory dit « MR brouillon uniquement » — une PR brouillon
   upstream est le livrable ; le fork sert de remote de branche. Aucun
   merge, aucun déploiement.

**Conséquence** : le clone de travail vit dans le workspace de la
conversation (`upstream-openhands/`, clone sparse) ; la branche
`agent/688e7f29-nest-subagent-conversations` est poussée sur le fork.

## D2 — Source de vérité du lien parent (2026-10-10)

**Question** : nicher via `parent_conversation_id` (porté par l'enfant) ou
via `sub_conversation_ids` (porté par le parent) ?

**Hypothèse retenue** : `parent_conversation_id` sur l'enfant.

- Le champ est **déjà servi** par l'agent-server local v1.49.6 (vérifié :
  `GET /api/conversations/search` renvoie `parent_conversation_id` sur
  chaque item) et par le SDK (`openhands.sdk.conversation.request`).
- `sub_conversation_ids` est « derived from the server catalog; empty on
  webhook payloads » (doc du modèle SDK) et ne dit pas si l'enfant est
  chargé dans les pages courantes — la relation portée par l'enfant se
  prête au filtrage de liste plate.
- Le helper `isPlannerConversationOf` (plan-file.ts) montre le pattern :
  identité par tag/lien, jamais par position. Le mode `/plan` pose le tag
  `plannerparent` et est filtré de la liste ; les enfants lancés par le
  tool `launch_child_conversation` n'ont **pas** ce tag, donc ils
  s'affichent à plat — c'est exactement le bug signalé.

**Conséquence** : ajouter `parent_conversation_id` au wire frontend
(`DirectConversationInfo` + `AppConversation`) et nicher les enfants sous
leur parent dans le panneau. Les enfants dont le parent n'est pas chargé
(pagination) restent visibles à plat — jamais cachés.

## D3 — UX de nichage retenue (2026-10-10)

- Chevron sur la carte du parent (visible dès qu'au moins un enfant chargé
  référence ce parent), rotation 90° à l'ouverture.
- Enfants indentés sous le parent, triés par le champ de tri courant.
- Replié par défaut ; état par conversation dans le state local du panneau
  (même pattern que `collapsedGroupIds`), non persisté.
- Auto-dépliage quand l'enfant actif (conversation courante) est un enfant
  ou qu'un enfant tourne (RUNNING) — pour ne pas cacher une activité.
- Badge count sur le chevron quand replié.
- Le nichage s'applique en mode chronological ET dans les dossiers groupés
  (workspace/repo), via un helper partagé.

## D4 — Réalisations finales vs D3 (2026-10-10, fin d'implémentation)

- Auto-dépliage implémenté uniquement sur "enfant actif" (conversation
  courante). Le cas "un enfant tourne (RUNNING)" a été écarté : déplier
  automatiquement sur statut d'exécution provoquerait des dépliages
  sauvages pendant les pollings ; l'auto-dépliage par navigation suffit
  au besoin exprimé (ne jamais perdre la conversation active).
- Enfants nichés rendus via un composant dédié `SubConversationRow`
  (titre + statut + âge), pas une `ConversationCard` complète : les
  métadonnées riches restent sur le parent, la liste reste compacte.
  Ligne volontairement hors du `NavigationLink` parent (sinon le clic
  enfant naviguerait vers le parent — ancres imbriquées interdites).
- Mode groupé : le scan des relations parent/enfant se fait sur
  `groupedSourceConversations` (avant groupage) pour qu'un enfant isolé
  dans un autre worktree/dossier se niche quand même sous son parent
  (son slot plat disparaît de l'autre dossier).
- Section épinglée volontairement plate (`subConversations: []`) : zone
  d'accès rapide, le nichage y dupliquerait les enfants.
- Tests : helper unitaire (4) + chronologique (2) + groupé (1) =
  7 nouveaux, 458 verts au total sur conversation-panel + adapter.
- Orphelins (parent non chargé) : gardés au premier niveau dans les deux
  modes — règle "rien ne disparaît" du ticket.
