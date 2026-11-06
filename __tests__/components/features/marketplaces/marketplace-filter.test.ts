import { describe, expect, it } from "vitest";
import {
  buildFacetGroups,
  EMPTY_MARKETPLACE_FILTER,
  filterEntries,
  toggleFacet,
} from "#/components/features/marketplaces/marketplace-filter";
import type {
  CatalogEntry,
  CatalogKind,
  InstallState,
} from "#/types/marketplaces";

function entry(
  kind: CatalogKind,
  name: string,
  overrides: {
    state?: InstallState;
    runs_code?: boolean;
    tags?: string[];
  } = {},
): CatalogEntry {
  return {
    id: `${kind}:${name}@m`,
    kind,
    name,
    title: name,
    marketplace: "m",
    description: `${name} description`,
    tags: overrides.tags ?? [],
    version: null,
    runs_code: overrides.runs_code ?? false,
    also_in: [],
    install: {
      state: overrides.state ?? "not_installed",
      enabled: true,
      installed_ref: null,
      policy: "track",
    },
    requires: [],
    transport: null,
    icon: null,
  };
}

const ENTRIES = [
  entry("skill", "code-review", { state: "installed", tags: ["review"] }),
  entry("skill", "runbook", { state: "update_available", tags: ["ops"] }),
  entry("plugin", "deploy", { runs_code: true, tags: ["ops", "hooks"] }),
  entry("mcp", "ledger", { tags: ["finance"] }),
];

describe("marketplace filter", () => {
  it("narrows by tab, then search, then facets", () => {
    expect(
      filterEntries(ENTRIES, "skill", EMPTY_MARKETPLACE_FILTER),
    ).toHaveLength(2);
    expect(
      filterEntries(ENTRIES, "all", {
        ...EMPTY_MARKETPLACE_FILTER,
        query: "LEDG",
      }).map((e) => e.name),
    ).toEqual(["ledger"]);
    const opsOnly = toggleFacet(EMPTY_MARKETPLACE_FILTER, "tag", "ops");
    expect(filterEntries(ENTRIES, "all", opsOnly).map((e) => e.name)).toEqual([
      "runbook",
      "deploy",
    ]);
    expect(filterEntries(ENTRIES, "skill", opsOnly).map((e) => e.name)).toEqual(
      ["runbook"],
    );
  });

  it("counts an entry with an update as installed too", () => {
    const installed = toggleFacet(
      EMPTY_MARKETPLACE_FILTER,
      "status",
      "installed",
    );
    expect(filterEntries(ENTRIES, "all", installed).map((e) => e.name)).toEqual(
      ["code-review", "runbook"],
    );
    const status = buildFacetGroups(
      ENTRIES,
      "all",
      EMPTY_MARKETPLACE_FILTER,
    ).find((g) => g.id === "status");
    expect(status?.rows.map((r) => [r.value, r.count])).toEqual([
      ["installed", 2],
      ["update_available", 1],
      ["not_installed", 2],
    ]);
  });

  it("ORs within a group and ANDs across groups", () => {
    let filter = toggleFacet(EMPTY_MARKETPLACE_FILTER, "tag", "review");
    filter = toggleFacet(filter, "tag", "finance");
    expect(filterEntries(ENTRIES, "all", filter).map((e) => e.name)).toEqual([
      "code-review",
      "ledger",
    ]);
    filter = toggleFacet(filter, "content", "runs_code");
    expect(filterEntries(ENTRIES, "all", filter)).toEqual([]);
    // Toggling again removes the facet.
    filter = toggleFacet(filter, "content", "runs_code");
    expect(filterEntries(ENTRIES, "all", filter)).toHaveLength(2);
  });

  it("builds tag rows from the current tab and keeps a selected tag visible", () => {
    const groups = buildFacetGroups(
      ENTRIES,
      "plugin",
      EMPTY_MARKETPLACE_FILTER,
    );
    const tags = groups.find((g) => g.id === "tag");
    expect(tags?.rows.map((r) => r.label)).toEqual(["hooks", "ops"]);

    const withFinance = toggleFacet(EMPTY_MARKETPLACE_FILTER, "tag", "finance");
    const pluginTags = buildFacetGroups(ENTRIES, "plugin", withFinance).find(
      (g) => g.id === "tag",
    );
    expect(pluginTags?.rows.find((r) => r.value === "finance")).toMatchObject({
      checked: true,
      count: 0,
    });

    // A selected tag already among the top rows is not listed twice.
    const withOps = toggleFacet(EMPTY_MARKETPLACE_FILTER, "tag", "ops");
    const opsRows = buildFacetGroups(ENTRIES, "all", withOps).find(
      (g) => g.id === "tag",
    );
    expect(opsRows?.rows.filter((r) => r.value === "ops")).toHaveLength(1);
  });
});
