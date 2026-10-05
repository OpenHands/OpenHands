# Canvas color themes

Canvas has three rendering systems: scoped CSS/HeroUI for the interface, Prism
and Monaco for code, and xterm for terminal output. A palette must cover all of
them; changing the page background alone is not a complete theme.

## Adding a palette

1. Add a definition under `color-theme/definitions/`, including its appearance,
   grey scale, HeroUI HSL channels, and semantic token overrides.
2. Add the key to `ColorThemeKey` and register the definition in
   `color-theme/definitions/index.ts`. The settings menu is generated from this
   registry.
3. Select a Prism palette in `syntax-highlighter-themes.ts`, a Monaco palette in
   `file-diff-viewer.tsx`, and all 16 ANSI colors in `terminal-themes.ts`.
   These mappings are exhaustive so new themes cannot silently inherit a dark
   code or terminal palette.
4. Run the theme tests and inspect settings, portalled menus/modals, chat
   Markdown, inline code, diffs, and ANSI output. Check disabled, selected,
   hover, and keyboard-focus states, then switch back to every dark palette.

## Ownership and component colors

- `setColorTheme` is the selection/persistence entry point. React consumers use
  `useColorTheme`; do not keep another copy of selected-theme state.
- The stylesheet owns theme defaults. `AgentServerUIRoot` owns its React
  attributes and preserves caller `styleOverrides` / `style` inline. The
  runtime must never erase or rewrite those caller-owned properties.
- Runtime CSS includes only palette overrides, not a second copy of base
  defaults. Omitted tokens continue to resolve from the base/host stylesheet.
- Embedded roots use the default appearance until a palette has actually been
  applied; a stored Canvas preference alone must not select an unstyled light
  wrapper. The application owns loading and applying its saved preference.
- The head bootstrap uses the same CSS generator before hydration, including
  when storage is unavailable or contains an obsolete key.
- Use surface/foreground tokens for ordinary content. `contrast` is the
  high-emphasis ink (white in dark palettes, dark in light palettes); its
  paired `contrast-foreground` is for inverse pills and tooltips. Opacity and
  state variants such as `hover:bg-contrast/10` retain their stated values.
- Literal white is reserved for fixed-color surfaces (for example the blue
  Plan control, an image-lightbox scrim, or a document preview). Do not globally
  redefine white or add theme-specific selectors that reinterpret utilities.
- SVG recoloring must preserve transparent negative-space paths.

CSS custom properties cannot be passed as colors to Monaco or xterm. Those
renderers require concrete colors; xterm's main foreground is resolved from
the mounted scope so embedding overrides still work.
