import { beforeEach, describe, expect, it } from "vitest";
import { REPO_SUGGESTIONS } from "#/utils/suggestions/repo-suggestions";
import {
  MACROS_STORAGE_KEY,
  createDefaultMacros,
  unusedCatalogEntries,
} from "#/utils/macros";
import { useMacrosStore } from "#/stores/macros-store";

describe("macros store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useMacrosStore.setState({ macros: createDefaultMacros() });
  });

  it("seeds the four chat-menu macros by default", () => {
    const { macros } = useMacrosStore.getState();

    expect(macros.map((macro) => macro.id)).toEqual([
      "INCREASE_TEST_COVERAGE",
      "FIX_README",
      "AUTO_MERGE_PRS",
      "CLEAN_DEPENDENCIES",
    ]);
    expect(macros[0]?.prompt).toBe(REPO_SUGGESTIONS.INCREASE_TEST_COVERAGE);
  });

  it("adds a custom macro and persists it", () => {
    useMacrosStore.getState().addMacro({
      title: "Triage bugs",
      prompt: "List open bugs and propose a fix order.",
    });

    const { macros } = useMacrosStore.getState();
    const custom = macros.find((macro) => macro.title === "Triage bugs");
    expect(custom?.prompt).toBe("List open bugs and propose a fix order.");
    expect(custom?.catalogId).toBeUndefined();

    const persisted = JSON.parse(
      window.localStorage.getItem(MACROS_STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.macros).toEqual(macros);
  });

  it("updates and deletes a macro", () => {
    const id = useMacrosStore.getState().macros[0]!.id;

    useMacrosStore.getState().updateMacro(id, {
      title: "More tests",
      prompt: "Add one unit test.",
    });
    expect(useMacrosStore.getState().macros[0]).toMatchObject({
      id,
      title: "More tests",
      prompt: "Add one unit test.",
    });

    useMacrosStore.getState().removeMacro(id);
    expect(
      useMacrosStore.getState().macros.find((macro) => macro.id === id),
    ).toBeUndefined();
  });

  it("adds unused catalog macros without duplicating ones already present", () => {
    expect(
      unusedCatalogEntries(useMacrosStore.getState().macros).map(
        (entry) => entry.id,
      ),
    ).toEqual(["ADD_DOCS", "ADD_DOCKERFILE"]);

    useMacrosStore.getState().addFromCatalog("ADD_DOCS");
    useMacrosStore.getState().addFromCatalog("ADD_DOCS");

    const docs = useMacrosStore
      .getState()
      .macros.filter((macro) => macro.catalogId === "ADD_DOCS");
    expect(docs).toHaveLength(1);
    expect(docs[0]?.prompt).toBe(REPO_SUGGESTIONS.ADD_DOCS);
    expect(
      unusedCatalogEntries(useMacrosStore.getState().macros).map(
        (entry) => entry.id,
      ),
    ).toEqual(["ADD_DOCKERFILE"]);
  });
});
