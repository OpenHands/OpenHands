<a name="readme-top"></a>

<div align="center">
  <h1 align="center" style="border-bottom: none">🌌 AURORA AI</h1>
  <p align="center">
    <strong>La plateforme souveraine de développement IA de nouvelle génération.</strong>
  </p>
  <p align="center">
    Combinant la puissance des agents autonomes d'OpenHands et l'excellence de l'environnement Code - OSS.
  </p>
</div>

<hr>

Aurora AI est une plateforme web d'intelligence artificielle avancée conçue pour les équipes de développement modernes. Elle fusionne :

1. **La puissance d'OpenHands** comme moteur d'agent IA autonome (gestion des tâches, exécution, outil de commande).
2. **L'ergonomie de Code - OSS / VS Code** (éditeur Monaco, explorateur, terminaux multiples).
3. **Une interface utilisateur propriétaire haut de gamme** (Next.js, Tailwind CSS, Radix UI, Framer Motion) offrant une expérience "Copilot" centralisée.

---

## 🚀 DÉMARRAGE RAPIDE (QUICKSTART)

Pour exécuter Aurora AI localement sur votre machine de développement :

### Prérequis
- **Node.js** v22.12.x ou supérieur
- **npm** v10.x ou supérieur
- **uv** (pour l'installation automatique du serveur d'agent en arrière-plan)

### Cloner et Installer
```bash
# 1. Cloner notre dépôt souverain
git clone https://github.com/Frankenstein-dev197/OpenHands.git
cd OpenHands

# 2. Installer les dépendances
npm ci

# 3. Lancer l'environnement de développement complet (Vite, Ingress, Agent, Automations)
npm run dev
```

Une fois démarré, accédez à la plateforme à l'adresse [http://localhost:8000](http://localhost:8000).

---

## 🏛️ ARCHITECTURE TECHNIQUE

L'architecture d'Aurora AI repose sur deux couches robustes :

```
               UTILISATEUR (AURORA AI UI)
                     |
                     v
           CENTRE DE CONTRÔLE IA
              (Chat UI Copilot)
                     |
         +-----------+-----------+
         |           |           |
         v           v           v
    ÉDITEUR MONACO TERMINAL   NAVIGATEUR
         |           |           |
         +-----------+-----------+
                     |
                     v
             COORDINATEUR AGENT
           (OpenHands SDK Engine)
```

- **Frontend Propriétaire (Aurora UI)** : Développé en Next.js, React, Tailwind CSS et Radix UI. Il gère l'historique de chat avec streaming, le panneau des tâches, l'affichage de diffs Monaco en temps réel, et le workflow d'approbation (Human-in-the-Loop).
- **Moteur d'Agent (OpenHands Agent Server)** : Gère les sandboxes Docker, le cycle de vie de l'agent, et la communication bidirectionnelle en WebSocket.

---

## 🔒 WORKFLOW D'APPROBATION ET DE SÉCURITÉ

Avant chaque opération critique (par exemple la modification d'un fichier de configuration ou l'exécution de commandes sensibles), l'agent présente une demande claire à l'utilisateur :

- **Approuver (✔)** : L'action est exécutée immédiatement dans la sandbox.
- **Modifier (✏)** : L'utilisateur ajuste la commande ou le code proposé avant exécution.
- **Refuser (✖)** : L'action est avortée et l'agent s'adapte en conséquence.

---

## 📜 LICENCES ET CONDITIONS D'UTILISATION

Ce projet réutilise des composants open source conformément à leurs licences respectives :
- **OpenHands / OpenDevin** (MIT License) : Copyright (c) 2025 OpenHands contributors.
- **Code - OSS / Monaco Editor** (MIT License) : Copyright (c) Microsoft Corporation.

Pour plus de détails sur les choix d'architecture, l'audit de licence et les règles d'intégration, consultez le document [AUDIT_ARCHITECTURE.md](./AUDIT_ARCHITECTURE.md).
