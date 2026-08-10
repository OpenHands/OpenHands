<a name="readme-top"></a>

<div align="center">
  <h1 align="center" style="border-bottom: none">🧠 CORTEX</h1>
  <p align="center">
    <strong>La plateforme souveraine d'orchestration agentique de nouvelle génération.</strong>
  </p>
  <p align="center">
    Construite au-dessus du socle agentique d'OpenHands et de l'environnement Code - OSS.
  </p>
</div>

<hr>

Cortex est une plateforme de développement agentique avancée combinant :

1. **Le moteur OpenHands** : Assure l'infrastructure agentique, le runtime, l'exécution des outils, la sandbox sécurisée, les terminaux, l'accès au workspace et le support des serveurs **Model Context Protocol (MCP)**.
2. **La couche Cortex** : Fournit notre nouvelle couche produit, notre identité, notre UX, nos compétences d'orchestration (Skills) et nos **intégrations MCP** encapsulées de manière transparente.

---

## 🏛️ ARCHITECTURE FONDAMENTALE

Cortex respecte scrupuleusement la distinction suivante :
- **OpenHands** = moteur / infrastructure de communication de bas niveau (y compris l'interface client-serveur MCP).
- **Cortex** = produit / couche d'orchestration / expérience utilisateur / compétences unifiées de haut niveau (Cortex Skills Dashboard & Integrations).

```
                    CORTEX (EXPÉRIENCE UTILSATEUR)
                               │
       ├── Cortex UI & Chat (Copilot UI)
       ├── Cortex Orchestrator (Planification & Abstraction)
       └── Cortex Skills & Integrations (Abstractions de serveurs MCP et compétences)
                               │
                               ▼
                  OPENHANDS CORE (INFRASTRUCTURE)
                               │
       ├── Agent Runtime & Tools (Terminal, Browser, Workspace)
       └── MCP Host Core (Protocoles de communication stdio/SSE)
```

---

## 🚀 DÉMARRAGE RAPIDE (QUICKSTART)

Pour démarrer et développer sur la plateforme Cortex :

### Prérequis
- **Node.js** v22.12.x ou supérieur
- **npm** v10.x ou supérieur
- **uv** (pour la virtualisation automatique des dépendances python de l'agent)

### Démarrage local
```bash
# 1. Cloner notre dépôt Cortex
git clone https://github.com/Frankenstein-dev197/OpenHands.git
cd OpenHands

# 2. Installer les dépendances NPM
npm ci

# 3. Lancer l'intégralité de la stack (Vite + Ingress + Agent Server + Automations)
npm run dev
```

Accédez à l'interface Cortex à l'adresse [http://localhost:8000](http://localhost:8000).

---

## 📜 LICENCES ET CONFORMITÉ

Cortex est construit sur des composants haut de gamme sous licence **MIT** :
- **OpenHands / OpenDevin** (MIT License) : Copyright (c) 2025 OpenHands contributors.
- **Code - OSS / Monaco Editor** (MIT License) : Copyright (c) Microsoft Corporation.

Le détail complet de l'audit de licence et de la séparation d'architecture est documenté dans [AUDIT_ARCHITECTURE.md](./AUDIT_ARCHITECTURE.md).
