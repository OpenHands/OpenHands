# Ruckus — Control Desk

Ruckus gives the existing Agent Canvas frontend a distinct working environment: aubergine housing, a warm paper conversation area, acid-yellow controls, a dimensional R mark, and a labeled tool dock. This change adds no backend capabilities or product workflows.

## Design reference

- Accepted concept: [approved-concept.png](approved-concept.png), 1536 × 1024.
- Implemented workspace: [desktop-conversation.png](desktop-conversation.png).
- Home: [desktop-home.png](desktop-home.png).
- Phone: [mobile-home.png](mobile-home.png), [mobile-navigation.png](mobile-navigation.png).

The screenshots show the repository's MSW fixtures. The disconnected-agent and missing-LLM states are real preview limitations, not successful live runs. Fixture report names and text remain unchanged.

## Where the existing controls live

| Existing surface                                                            | Ruckus location                                               |
| --------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Conversations, folders, tags, filters, pin/archive/context actions          | Left conversation index                                       |
| New conversation and workspace selection                                    | New Chat and the home composer                                |
| Automations, templates, dashboard, run history                              | Automate in the top navigation; existing internal navigation  |
| MCP servers, skills, plugins, extensions                                    | Customize in the top navigation; existing internal navigation |
| Extension-contributed pages                                                 | Top navigation, horizontally scrollable when needed           |
| Command search and shortcuts                                                | Header search; mobile drawer search                           |
| Backend selection, connection health, add/manage backends                   | Header selector; mobile drawer; collapsed-rail shortcuts      |
| Agent, LLM, context, condenser, verification, application, secrets settings | Header settings control; existing settings navigation         |
| Chat messages, artifacts, attachments, approvals, execution controls        | Paper conversation area and composer                          |
| Files, commits, planner, terminal, browser, usage, conditional task list    | Resizable right panel and bottom tool dock                    |
| Tool pinning and overflow                                                   | Dock menu, opening upward near the viewport bottom            |
| Conversation metadata and Git actions                                       | Conversation heading                                          |
| Getting-started checklist and version notices                               | Conversation index footer                                     |
| Mobile navigation and tools                                                 | Existing drawers, with Ruckus styling                         |

Pin-as-home behavior, routes, API consumers, analytics, feature flags, persisted preferences, and tool availability remain owned by their existing implementations. Standalone library consumers keep the sidebar layout by default; the app opts into `Sidebar layout="desk"`.

## Visual system

- Housing: `#19131F`; panel: `#251B2D`; paper: `#F3F0E8`; ink: `#251B2D`; acid: `#D9F266`.
- Outfit for interface text and the heavy wordmark; IBM Plex Mono for small index labels. Existing editor typography is retained.
- Custom SVG brand and duotone navigation/tool glyphs; integration and provider logos retain their identities.
- Tight 4–6px corners on principal controls, restrained borders and shallow shadows.
- Small brand lift on hover and a gentle movement tied to an actually running conversation. Reduced-motion preferences disable the decorative movement.
- Existing theme selections remain available. Ruckus is the new default for an unset theme preference.

## Fidelity review

The approved concept and final desktop capture were both opened with `view_image` in the final QA pass, at the reference's 1536 × 1024 viewport. The implementation was visually verified against the accepted design for its principal layout, palette, brand treatment, and tool organization. The intentional differences below preserve the real application's features and data.

| Comparison          | Concept evidence                                            | Render evidence / resolution                                                                                                                                 |
| ------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Layout              | Horizontal navigation, left index, light chat, dark tools   | Same hierarchy in `desktop-conversation.png`; current resizable panel proportions remain user-controlled                                                     |
| Palette             | Near-black plum, warm white, yellow-green selections        | Dedicated dark theme plus locally scoped paper tokens; corrected inherited control colors and integration-logo contrast                                      |
| Typography          | Heavy RUCKUS wordmark, compact labels, legible working text | Heavy Outfit wordmark and home title; 14px navigation and 15px message text; existing editor font retained                                                   |
| Brand and icons     | Dimensional R and small geometric tool glyphs               | Code-native SVG R and custom navigation/tool icons; no raster screenshot used as interactive UI                                                              |
| Tool dock           | Tools along bottom of right pane                            | Labeled dock; active tool in acid; overflow opens upward instead of off-screen                                                                               |
| Containers          | Continuous working surfaces with restrained borders         | Paper chat and dark tools share the shell; dense existing settings/catalog components retain their structure                                                 |
| Responsive behavior | Desktop concept only                                        | Checked 1536 × 1024, 842 × 724, and 390 × 844; document width equals viewport width in each; mobile drawer remains usable                                    |
| Copy and states     | Illustrative conversation, file tree, model, status footer  | Retained real translations, message components, fixture data, alerts, and existing model controls. No invented status footer or mock success state was added |

Above-the-fold copy review: product title, logo label, and onboarding title use Ruckus in all 15 existing locales. Existing labels such as “New Chat,” “Automate,” “Planner,” and “Commits” are intentionally retained rather than replaced with the concept image's illustrative labels. References to the actual OpenHands agent, Cloud service, public community, telemetry recipient, and upstream package/version remain accurate. The generated concept's sample message text, timestamps, model name, and repository contents were not transplanted into the product.

## Verification

Browser checks used the Codex in-app browser and the repository's mock frontend. Verified home, conversation rendering, file rich/plain switching, file-tree visibility, tool-dock overflow, settings navigation, Customize, Automate, command search, and mobile drawer navigation. Captures use the browser screenshot API; desktop and phone layouts were inspected visually.

Automated validation passed: 643 test files with 5,573 passing tests (4 skipped and 7 TODO), focused navigation/menu/title regressions, lint/typechecking, translation completeness, application build, and library build. Lint retains one pre-existing unused-disable warning in `use-local-git-info.ts`. The full suite used `TZ=UTC` to avoid an existing epoch-date assertion that otherwise formats as December 31, 1969 in America/Chicago. This work does not establish live LLM, authenticated Cloud, deployment, or real backend acceptance.

Run `npm run dev:mock` for fixture-backed visual development, or use the repository's documented real-stack launch commands for agent execution.
