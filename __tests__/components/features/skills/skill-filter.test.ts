import { describe, expect, it } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo } from "#/types/settings";
import {
  applySkillFilters,
  buildSkillFacetGroups,
  clearSkillFilterFacets,
  countActiveFilters,
  EMPTY_SKILL_FILTER_STATE,
  parseSkillFilterState,
  toggleSkillFilterValue,
  toSkillFilterSearchParams,
  type SkillFilterState,
} from "#/components/features/skills/skill-filter";

function buildSkill(overrides: Partial<SkillInfo> = {}): SkillInfo {
  return {
    name: "deno",
    type: "knowledge",
    source: "public",
    category: "environment",
    description: "Initialize and work with Deno projects.",
    triggers: ["deno"],
    ...overrides,
  };
}

/** Public env skill, public writing skill, personal repo skill. */
function buildMixedSkills(): SkillInfo[] {
  return [
    buildSkill({ name: "deno", category: "environment", type: "knowledge" }),
    buildSkill({
      name: "prd",
      category: "writing",
      type: "knowledge",
      description: "Generate a Product Requirements Document.",
      triggers: [],
    }),
    buildSkill({
      name: "house-rules",
      category: null,
      type: "repo",
      source: "/home/dev/.agents/skills/house-rules/SKILL.md",
      description: "Local conventions.",
      triggers: [],
    }),
  ];
}

function stateWith(
  overrides: Partial<SkillFilterState> = {},
): SkillFilterState {
  return { ...EMPTY_SKILL_FILTER_STATE, ...overrides };
}

describe("parseSkillFilterState", () => {
  it("reads every facet plus the query", () => {
    const state = parseSkillFilterState(
      new URLSearchParams(
        "q=deno&source=personal&source=public&category=environment&type=repo&state=disabled",
      ),
    );

    expect(state.query).toBe("deno");
    expect([...state.sources]).toEqual(["personal", "public"]);
    expect([...state.categories]).toEqual(["environment"]);
    expect([...state.types]).toEqual(["repo"]);
    expect([...state.states]).toEqual(["disabled"]);
  });

  it.each([
    ["source", "source=nonsense"],
    ["category", "category=code-hostig"],
    ["type", "type=bogus"],
    ["state", "state=maybe"],
  ])("drops unknown %s values instead of erroring", (_label, search) => {
    const state = parseSkillFilterState(new URLSearchParams(search));
    expect(countActiveFilters(state)).toBe(0);
  });

  it("round-trips through toSkillFilterSearchParams in canonical order", () => {
    const search =
      "q=deno&source=project&source=public&category=environment&category=writing";
    const state = parseSkillFilterState(new URLSearchParams(search));

    expect(toSkillFilterSearchParams(state).toString()).toBe(
      new URLSearchParams(search).toString(),
    );
  });

  it("omits empty facets so an unfiltered page has a bare URL", () => {
    expect(toSkillFilterSearchParams(EMPTY_SKILL_FILTER_STATE).toString()).toBe(
      "",
    );
  });
});

describe("applySkillFilters", () => {
  const skills = buildMixedSkills();
  const disabled = new Set(["prd"]);

  it("ORs values within a group", () => {
    const result = applySkillFilters(
      skills,
      disabled,
      stateWith({ categories: new Set(["environment", "writing"]) }),
    );
    expect(result.map((s) => s.name)).toEqual(["deno", "prd"]);
  });

  it("ANDs across groups", () => {
    const result = applySkillFilters(
      skills,
      disabled,
      stateWith({
        categories: new Set(["environment", "writing"]),
        states: new Set(["disabled"]),
      }),
    );
    expect(result.map((s) => s.name)).toEqual(["prd"]);
  });

  it("treats a category-less skill as other", () => {
    const result = applySkillFilters(
      skills,
      disabled,
      stateWith({ categories: new Set(["other"]) }),
    );
    expect(result.map((s) => s.name)).toEqual(["house-rules"]);
  });

  it("matches the query against name, description, and triggers", () => {
    expect(
      applySkillFilters(
        skills,
        disabled,
        stateWith({ query: "Requirements" }),
      ).map((s) => s.name),
    ).toEqual(["prd"]);
    expect(
      applySkillFilters(skills, disabled, stateWith({ query: "deno" })).map(
        (s) => s.name,
      ),
    ).toEqual(["deno"]);
  });

  it("returns everything when no facet is selected", () => {
    expect(
      applySkillFilters(skills, disabled, EMPTY_SKILL_FILTER_STATE),
    ).toHaveLength(3);
  });
});

describe("buildSkillFacetGroups", () => {
  it("hides a group whose values cannot discriminate", () => {
    const publicOnly = [
      buildSkill({ name: "deno", category: "environment" }),
      buildSkill({ name: "uv", category: "environment" }),
    ];

    const groups = buildSkillFacetGroups(
      publicOnly,
      new Set(),
      EMPTY_SKILL_FILTER_STATE,
    );

    // Every skill is public, knowledge-typed, environment-categorised and
    // enabled, so no group has two non-empty values.
    expect(groups).toEqual([]);
  });

  it("keeps a group visible when its only selected value has no matches for this user", () => {
    // Reproduces sharing a `?source=project` URL with someone who has no project skills: hiding the group would enforce the filter with no row left to undo it.
    const publicOnly = [
      buildSkill({ name: "deno", source: "public" }),
      buildSkill({ name: "uv", source: "public" }),
    ];

    const groups = buildSkillFacetGroups(
      publicOnly,
      new Set(),
      stateWith({ sources: new Set(["project"]) }),
    );

    const source = groups.find((g) => g.id === "source");
    expect(source).toBeDefined();
    expect(source!.rows.find((r) => r.value === "project")).toMatchObject({
      checked: true,
      count: 0,
    });
  });

  it("shows groups once a second value exists", () => {
    const groups = buildSkillFacetGroups(
      buildMixedSkills(),
      new Set(["prd"]),
      EMPTY_SKILL_FILTER_STATE,
    );

    expect(groups.map((g) => g.id)).toEqual([
      "state",
      "source",
      "category",
      "type",
    ]);
  });

  it("labels rows with i18n keys rather than translated text", () => {
    const groups = buildSkillFacetGroups(
      buildMixedSkills(),
      new Set(["prd"]),
      EMPTY_SKILL_FILTER_STATE,
    );
    const category = groups.find((g) => g.id === "category")!;

    expect(category.labelKey).toBe(I18nKey.SETTINGS$SKILLS_FACET_CATEGORY);
    expect(category.rows.map((r) => r.value)).toEqual([
      "environment",
      "writing",
      "other",
    ]);
    expect(category.rows[0]!.labelKey).toBe(
      I18nKey.SETTINGS$SKILLS_CATEGORY_ENVIRONMENT,
    );
  });

  it("counts rows against other groups but not the group's own selection", () => {
    const groups = buildSkillFacetGroups(
      buildMixedSkills(),
      new Set(["prd"]),
      stateWith({ states: new Set(["enabled"]) }),
    );

    const category = groups.find((g) => g.id === "category")!;
    const byValue = Object.fromEntries(
      category.rows.map((r) => [r.value, r.count]),
    );

    // The `state` selection narrows Category, dropping the disabled `prd`.
    expect(byValue).toEqual({ environment: 1, writing: 0, other: 1 });

    // The `state` group ignores its own selection, so both rows keep counts.
    const state = groups.find((g) => g.id === "state")!;
    expect(state.rows.map((r) => r.count)).toEqual([2, 1]);
  });

  it("keeps a different group's visibility stable while one facet narrows", () => {
    const skills = buildMixedSkills();
    const disabled = new Set(["prd"]);

    const narrowed = buildSkillFacetGroups(
      skills,
      disabled,
      stateWith({ categories: new Set(["environment"]) }),
    );

    // Only `deno` survives, leaving Source and Type one value each among the matches; visibility comes from the raw list, so both groups still render.
    expect(narrowed.map((g) => g.id)).toEqual(
      expect.arrayContaining(["source", "type"]),
    );
  });

  it("keeps group visibility stable while a search query narrows results", () => {
    const skills = buildMixedSkills();
    const disabled = new Set(["prd"]);
    const unfiltered = buildSkillFacetGroups(
      skills,
      disabled,
      EMPTY_SKILL_FILTER_STATE,
    );

    // "Requirements" matches only `prd`, so a query-filtered denominator would collapse every group to one value.
    const narrowed = buildSkillFacetGroups(
      skills,
      disabled,
      stateWith({ query: "Requirements" }),
    );

    expect(narrowed.map((g) => g.id)).toEqual(unfiltered.map((g) => g.id));
  });

  it("disables a zero-count row unless it is checked", () => {
    const groups = buildSkillFacetGroups(
      buildMixedSkills(),
      new Set(["prd"]),
      stateWith({ states: new Set(["enabled"]) }),
    );
    const rows = groups.find((g) => g.id === "category")!.rows;

    expect(rows.find((r) => r.value === "writing")).toMatchObject({
      count: 0,
      checked: false,
      disabled: true,
    });
  });
});

describe("toggleSkillFilterValue", () => {
  it("adds then removes a value without mutating the input", () => {
    const initial = EMPTY_SKILL_FILTER_STATE;
    const added = toggleSkillFilterValue(initial, "category", "writing");
    const removed = toggleSkillFilterValue(added, "category", "writing");

    expect([...added.categories]).toEqual(["writing"]);
    expect([...removed.categories]).toEqual([]);
    expect([...initial.categories]).toEqual([]);
  });

  it("ignores a value that is not legal for the group", () => {
    const next = toggleSkillFilterValue(
      EMPTY_SKILL_FILTER_STATE,
      "type",
      "nonsense",
    );
    expect(countActiveFilters(next)).toBe(0);
  });

  it.each([
    ["source", "sources", "project"],
    ["state", "states", "disabled"],
  ] as const)(
    "toggles a %s value into its own set, leaving the rest empty",
    (groupId, key, value) => {
      const next = toggleSkillFilterValue(
        EMPTY_SKILL_FILTER_STATE,
        groupId,
        value,
      );

      // Catches a branch writing to the wrong field: either the intended set stays empty, or an unintended one gains the value.
      const collections = ["sources", "categories", "types", "states"] as const;
      for (const collection of collections) {
        const expected = collection === key ? [value] : [];
        expect([...next[collection]]).toEqual(expected);
      }
    },
  );
});

describe("clearSkillFilterFacets and countActiveFilters", () => {
  it("clears facets but keeps the query", () => {
    const state = stateWith({
      query: "deno",
      categories: new Set(["writing"]),
      types: new Set(["repo"]),
    });

    const cleared = clearSkillFilterFacets(state);

    expect(cleared.query).toBe("deno");
    expect(countActiveFilters(cleared)).toBe(0);
  });

  it("does not count the query as a filter", () => {
    expect(countActiveFilters(stateWith({ query: "deno" }))).toBe(0);
    expect(
      countActiveFilters(
        stateWith({ query: "deno", categories: new Set(["writing"]) }),
      ),
    ).toBe(1);
  });
});
