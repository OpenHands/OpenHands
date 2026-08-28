// HeroUI v3 uses @import "@heroui/styles" in tailwind.css instead of the v2
// heroui() Tailwind plugin. Theme overrides are applied via CSS custom
// properties (see color-themes.ts for the runtime theme-switching system).
//
// The file is kept as a module so the resolve-affected-tests.mjs trigger
// still works — removing it would change the E2E test selection behaviour.
export default {};
