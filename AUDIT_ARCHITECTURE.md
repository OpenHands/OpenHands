# 🧠 SPÉCIFICATION D'ARCHITECTURE & AUDIT : PLATEFORME CORTEX

Ce document décrit l'intégration de la puissance de l'agent autonome de **OpenHands** avec l'environnement utilisateur haut de gamme et les couches d'orchestration de **CORTEX**.

La règle fondamentale de notre plateforme est :
> « **OpenHands** fournit le moteur agentique et ses fonctionnalités existantes. **Cortex** fournit notre nouvelle couche produit, notre identité, notre UX, nos compétences et nos fonctionnalités supplémentaires. »

---

## 1. STRATÉGIE DE COUPLAGE ET BOUNDARIES

Cortex se positionne comme la couche d'orchestration supérieure, encapsulant les capacités de bas niveau d'OpenHands sans perturber leur fonctionnement éprouvé.

```
+-----------------------------------------------------------------------+
|                                CORTEX                                 |
|                                                                       |
|   +---------------+   +---------------+   +-----------------------+   |
|   |   Cortex UI   |   |  Cortex Chat  |   |   Cortex Workbench    |   |
|   +---------------+   +---------------+   +-----------------------+   |
|   | Cortex Skills |   | Cortex Memory |   |  Cortex Orchestrator  |   |
|   +---------------+   +---------------+   +-----------------------+   |
|   |  Workflows    |   |  Permissions  |   | Projects/Integrations |   |
|   +---------------+   +---------------+   +-----------------------+   |
+-----------------------------------+-----------------------------------+
                                    |
                                    v
+-----------------------------------+-----------------------------------+
|                           OPENHANDS CORE                              |
|                                                                       |
|   +-----------------------+  +------------------+  +--------------+   |
|   |     Agent Runtime     |  |      Tools       |  |   Sandbox    |   |
|   +-----------------------+  +------------------+  +--------------+   |
|   |       Terminal        |  |     Browser      |  |  Workspace   |   |
|   +-----------------------+  +------------------+  +--------------+   |
+-----------------------------------------------------------------------+
```

### 1.1 Rôles Respectifs
- **Moteur OpenHands (Agent Server / Core)** : Fournit le Runtime de l'agent, l'exécution des commandes, la gestion et la persistance des fichiers de workspace, les terminaux `xterm.js`, l'accès au navigateur interne, et les protocoles WebSocket/REST de communication temps réel.
- **Produit Cortex** : Gère l'expérience utilisateur globale, le design system, l'orchestration des flux de décision, les contrôles d'autorisation interactifs (Human-in-the-Loop), et l'extensibilité par compétences métier (Skills).

---

## 2. COMPOSANTS DE LA COUCHE CORTEX

La couche Cortex se décompose en modules spécialisés pour structurer proprement l'implémentation :

### 2.1 Cortex UI & Chat (Centre de Contrôle)
L'expérience utilisateur est profondément modernisée sous le design system **Cortex UI**.
- Le **Cortex Chat** agit comme le point d'entrée d'orchestration unique (inspiré de Copilot Chat).
- Les étapes de pensée interne des agents OpenHands sont enveloppées de manière non-intrusive (Collapsible Thinking) pour une lisibilité maximale.
- Les actions nécessitant des permissions (ex: modification de configuration, exécution de commandes critiques) passent par un composant d'approbation dédié **Cortex Permissions** avec boutons `Approuver`, `Modifier`, `Refuser`.

### 2.2 Cortex Orchestrator & Skills
L'orchestration Cortex permet de guider l'agent avant l'exécution finale :
- **Cortex Orchestrator** : Traduit les requêtes utilisateur de haut niveau, sélectionne les compétences appropriées, et configure le payload de conversation envoyé au moteur OpenHands.
- **Cortex Skills** : Une architecture extensible pour encapsuler les capacités spécialisées. Les compétences (ex: développement web, debugging de test, DevOps, déploiement) sont déclarées dynamiquement et associées au runtime sous forme d'outils complémentaires.

---

## 3. AUDIT DES LICENCES ET CONFORMITÉ

La réutilisation et la personnalisation d'OpenHands (sous licence MIT) et de Monaco Editor / Code-OSS (sous licence MIT) sont pleinement en conformité avec les obligations légales :
- Les notices de copyright d'origine de **OpenHands contributors** et de **Microsoft Corporation** sont conservées à la racine du projet et dans les dépendances applicables.
- Cortex constitue une couche d'abstraction propriétaire légalement distincte s'exécutant au-dessus des composants open-source.
