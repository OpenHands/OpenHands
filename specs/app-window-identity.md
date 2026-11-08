# App window identity

### AWI-001: Localized section titles

- [x] Matched route handles own section titles; unknown nested Settings and Automations URLs retain the `OpenHands` fallback on the not-found page.
- [x] `/` and `/conversations` share the localized New Chat title. Settings descendants inherit their section title. Customize, MCP, Skills, Plugins, and Apps use their existing page translations.
- [x] Automation routes use the admitted interface's sidebar label and fall back to `OpenHands` when that interface is unavailable. Registered extension pages retain their contribution title, including nested paths.
- [x] Conversation and panel routes retain their conversation title and execution-status emoji. Query refreshes, including renames, update the title. Loading and untitled conversations retain the stable `OpenHands` fallback.

### AWI-002: Installed-app launches and shortcuts

- [x] The manifest requests `navigate-existing` launches and preferred link handling. Browser support and the user's link-capture settings determine whether the browser honors these declarations. `navigate-existing` reloads the document at the target URL; unsaved form input can be lost. This change adds no navigation blocker or client-side launch consumer.
- [x] New conversation and Automations shortcuts have distinct destinations, resolve to real Canvas routes and existing icons relative to the manifest, and remain inside the explicit manifest scope at both the origin root and a deployment base path. The composer shortcut targets `/conversations`, avoiding the configurable redirect at `/`.
- [ ] Installation and an installed-window transition recording depend on the separate PWA foundation (#17890 / #17895). Its manifest link, app name, branded icons, and platform installation metadata remain outside this change. The prior temporary integration run is historical manual evidence, not proof that this branch alone is installable.
