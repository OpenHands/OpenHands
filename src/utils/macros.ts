import { REPO_SUGGESTIONS } from "#/utils/suggestions/repo-suggestions";

export type Macro = {
  id: string;
  title: string;
  prompt: string;
  catalogId?: string;
};

export type MacroCatalogEntry = {
  id: string;
  titleKey: string;
  prompt: string;
};

export const MACROS_STORAGE_KEY = "openhands-macros";

export const DEFAULT_MACRO_CATALOG_IDS = [
  "INCREASE_TEST_COVERAGE",
  "FIX_README",
  "AUTO_MERGE_PRS",
  "CLEAN_DEPENDENCIES",
] as const;

const CATALOG_TITLE_KEYS: Record<string, string> = {
  INCREASE_TEST_COVERAGE: "INCREASE_TEST_COVERAGE",
  FIX_README: "FIX_README",
  AUTO_MERGE_PRS: "AUTO_MERGE_PRS",
  CLEAN_DEPENDENCIES: "CLEAN_DEPENDENCIES",
  ADD_DOCS: "SUGGESTIONS$ADD_DOCS",
  ADD_DOCKERFILE: "SUGGESTIONS$ADD_DOCKERFILE",
};

export const MACRO_CATALOG: MacroCatalogEntry[] = Object.entries(
  REPO_SUGGESTIONS,
).map(([id, prompt]) => ({
  id,
  titleKey: CATALOG_TITLE_KEYS[id] ?? id,
  prompt,
}));

export function createMacroId(): string {
  return typeof crypto.randomUUID === "function"
    ? crypto.randomUUID()
    : `macro-${Date.now()}`;
}

export function createDefaultMacros(): Macro[] {
  return DEFAULT_MACRO_CATALOG_IDS.flatMap((id) => {
    const entry = MACRO_CATALOG.find((item) => item.id === id);
    if (!entry) {
      return [];
    }
    return [
      {
        id: entry.id,
        title: entry.titleKey,
        prompt: entry.prompt,
        catalogId: entry.id,
      },
    ];
  });
}

export function unusedCatalogEntries(macros: Macro[]): MacroCatalogEntry[] {
  const used = new Set(
    macros.flatMap((macro) => (macro.catalogId ? [macro.catalogId] : [])),
  );
  return MACRO_CATALOG.filter((entry) => !used.has(entry.id));
}

export function isMacro(value: unknown): value is Macro {
  if (!value || typeof value !== "object") {
    return false;
  }
  const macro = value as Macro;
  return (
    typeof macro.id === "string" &&
    macro.id.length > 0 &&
    typeof macro.title === "string" &&
    typeof macro.prompt === "string"
  );
}
