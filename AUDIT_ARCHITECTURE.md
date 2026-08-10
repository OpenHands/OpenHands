# 📑 AUDIT & ARCHITECTURE SPECIFICATION: AI DEVELOPMENT PLATFORM "AURORA AI"

Ce document d'architecture et d'audit décrit l'intégration de la puissance de l'agent autonome d'**OpenHands** avec l'ergonomie de développement de **Code - OSS (Visual Studio Code / Monaco)**, sous une nouvelle identité visuelle propriétaire appelée **Aurora AI**.

---

## 1. AUDIT DE LICENCE ET CONFORMITÉ

### 1.1 OpenHands / OpenDevin (MIT License)
- **Modèle de licence** : MIT (permet l'utilisation commerciale, la modification, la distribution et la sous-licence).
- **Contraintes** : Obligation d'inclure la notice de copyright originale et la mention de licence MIT dans toutes les copies ou parties substantielles du logiciel.
- **Réutilisation de code** : Nous conservons et encapsulons les composants d'agent (communication via `@openhands/typescript-client`, gestion des tâches, connexion WebSocket, terminaux de commande xterm, et visualiseur de navigateur).

### 1.2 Visual Studio Code / Code - OSS (MIT License)
- **Modèle de licence** : MIT pour Code - OSS, tandis que l'application officielle "VS Code" de Microsoft est propriétaire.
- **Composant réutilisé (Monaco Editor)** : Monaco Editor (le cœur de l'éditeur de VS Code) est entièrement open source sous licence MIT. Il est intégré via `@monaco-editor/react`.
- **Système d'onglets, panneaux et explorateur** : Réutilisés et adaptés en React, TypeScript et Tailwind CSS pour une intégration directe sans le surpoids de l'architecture Electron complète de VS Code.

---

## 2. ARCHITECTURE CONCEPTUELLE ET SYSTEM BOUNDARIES

```
                             AURORA AI (UTILISATEUR)
                                        |
                                        v
                            [ NOUVELLE INTERFACE IA ]
                      (Next.js, Tailwind, shadcn/ui, Radix)
                                        |
                 +----------------------+----------------------+
                 |                                             |
                 v                                             v
        [ CHAT UI COPILOT ]                        [ WORKSPACE DÉVELOPPEUR ]
   (Centre de contrôle de l'agent)                 - Explorateur de fichiers (Code-OSS style)
  Streaming, Outils, Plans, Actions                - Monaco Code Editor (Visualiseur de Diff)
                 |                                 - Terminaux d'Exécution (xterm.js)
                 |                                 - Navigateur web embarqué (Aperçu)
                 +----------------------+----------------------+
                                        |
                                        v
                             [ COUCHE AGENT IA ]
                      (WebSocket, REST API SDK clients)
                                        |
                                        v
                          [ OPENHANDS AGENT SERVER ]
                             (Sandbox / Workspace)
```

### 2.1 Frontières du Système (System Boundaries)
- **Aurora AI Frontend** : Responsable du rendu, du design system propriétaire, du Chat UI, de la gestion des onglets, du terminal de contrôle, des diffs de fichiers et de l'orchestration des flux d'approbation.
- **Agent Server Backend** : Responsable de l'exécution des commandes, du cycle de vie de l'agent IA, de la manipulation de la sandbox et de la virtualisation du navigateur.

---

## 3. NOUVEAU DESIGN SYSTEM ET IDENTITÉ ("AURORA AI")

Pour remplacer l'apparence générique par défaut d'OpenHands par une véritable plateforme souveraine, Aurora AI implémente les changements suivants :

- **Nom de marque** : Aurora AI (L'Aurore du développement assisté par l'IA).
- **Thème de couleur** : Un thème sombre moderne mêlant les tons d'espace profond (`Slate/Zinc` ultra-sombres), des accents d'aurore boréale (`Emerald` pour la réussite, `Violet/Indigo` pour l'IA, `Cyan` pour l'éditeur).
- **Navigation** : Rail latéral ultra-fin (Sidebar) avec icônes Lucide-React minimalistes, accès rapide au menu de commande, et un panneau de conversation compact.
- **Animations** : Framer Motion est utilisé pour l'apparition fluide des cartes d'action, le déploiement des étapes de pensée (Collapsible Thinking) et les transitions d'onglets.

---

## 4. CHAT UI : LE CENTRE DE CONTRÔLE COPILOT

La nouvelle interface de chat s'inspire du fonctionnement de **VS Code Copilot Chat**, consolidant tout le flux d'instructions et d'interventions en un flux centralisé :

- **Streaming Temps Réel** : Les réponses textuelles et les étapes de pensée de l'agent s'affichent au fil de l'eau.
- **États d'Exécution & Outils** : Chaque outil appelé par l'agent (ex: modification de fichier, commande Bash) est représenté sous forme de badge interactif cliquable pour voir le détail de l'action.
- **Pensée Collapsible (Thinking Blocks)** : Les étapes de raisonnement interne de l'agent (de type `<thinking>`) sont masquées par défaut derrière un accordéon élégant pour ne pas polluer la lecture.

---

## 5. L'IA TRAVAILLE DEVANT L'UTILISATEUR (REPRÉSENTATION VISUELLE)

La plateforme affiche de manière transparente chaque étape du travail de l'agent :

1. **Le Plan de l'Agent** : Un panneau ou onglet dédié répertorie la liste des tâches (Task List) en cours, complétées ou en attente.
2. **Commandes Bash** : Affichage en direct du terminal interactif (via `xterm.js`) exécutant les commandes de compilation, d'installation de dépendances, ou de tests.
3. **Fichiers Modifiés (Diffs)** : Ouverture automatique des fichiers modifiés dans un éditeur Monaco en mode Diff, montrant précisément l'avant/après de chaque modification de code effectuée par l'agent.

---

## 6. WORKFLOW D'APPROBATION ET DE CONTRÔLE (HUMAN-IN-THE-LOOP)

Pour les actions critiques, Aurora AI maintient l'utilisateur aux commandes :

- **Déclenchement** : Lorsqu'une action sensible est initiée (ex: écriture de fichier, exécution de commande système en mode confirmation, modification de configuration).
- **Rendu Visuel** :
  ```
  +--------------------------------------------------------------+
  |  ⚠️ Aurora AI souhaite exécuter cette action :                 |
  |  Modification du fichier : src/App.tsx                       |
  |                                                              |
  |  [ APPROUVER (✔) ]     [ MODIFIER (✏) ]     [ REFUSER (✖) ]   |
  +--------------------------------------------------------------+
  ```
- **États** : L'état d'approbation (En attente, Approuvé, Rejeté) est immédiatement sérialisé et reflété par des codes couleur précis dans l'historique du chat.

---

## 7. STACK FRONTEND & DIRECTIVES DE PRODUCTION

La stack s'appuie sur des outils performants pour garantir une qualité optimale en production :
- **Next.js & React 19** : Cadre applicatif performant et composable.
- **Tailwind CSS** : Style modulaire et responsive via des variables CSS thématiques.
- **shadcn/ui & Radix UI** : Accessibilité de premier plan et composants d'interface robustes (Dialog, Dropdown, Accordion).
- **Framer Motion** : Rendu dynamique et transitions professionnelles.
- **i18next** : Localisation complète en 15 langues avec typages stricts.
