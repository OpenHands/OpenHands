# App window identity

### AWI-001: Localized section titles

- [x] Home, Conversations, Automations (including nested routes), and Settings (including nested routes) identify their section in the document title using existing translations, followed by ` | OpenHands`. Client-side navigation and language changes update the title through `useAppTitle` and the existing root-layout title element.
- [x] Conversation and panel routes retain their conversation title and execution-status emoji. Query refreshes, including renames, update the title. Loading and untitled conversations retain the stable `OpenHands` fallback; unrelated routes also retain that fallback.

### AWI-002: Installed-app launches and shortcuts

- [x] The manifest requests `navigate-existing` launches and preferred link handling. Browser support and the user's link-capture settings determine whether the browser honors these declarations.
- [x] New conversation, Conversations, and Automations shortcuts resolve to real Canvas routes and existing icons relative to the manifest, both at the origin root and under a deployment base path. New conversation and Conversations both open the existing `/conversations` home/composer surface; neither follows the configurable redirect at `/`.
- [ ] Installed-window navigation, OS shortcuts, and window reuse are verified with the separate PWA installation foundation (#17890). That foundation, a service worker, notifications, and Electron behavior are outside this change.
