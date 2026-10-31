import React from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { act, screen, waitFor, within, render } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import {
  SKILL_CARD_PILL_CLASS,
  SkillCardPillRow,
} from "#/components/features/skills/skill-card-pill-row";
import { I18nKey } from "#/i18n/declaration";
import { SkillFacetRail } from "#/components/features/skills/skill-facet-rail";
import {
  type SkillFacetGroup,
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
import { type SkillInfo } from "#/types/settings";
import { SkillFiltersModal } from "#/components/features/skills/skill-filters-modal";

describe("SkillCardPillRow", () => {
  const observedCallbacks: ResizeObserverCallback[] = [];

  beforeEach(() => {
    observedCallbacks.length = 0;
    vi.stubGlobal(
      "ResizeObserver",
      class {
        constructor(cb: ResizeObserverCallback) {
          observedCallbacks.push(cb);
        }

        observe() {
          const cb = observedCallbacks[observedCallbacks.length - 1];
          cb?.([], this as unknown as ResizeObserver);
        }

        disconnect() {}

        unobserve() {}
      },
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function stubWidths(containerWidth: number, pillWidth: number) {
    const row = screen.getByTestId("skill-triggers-test");
    Object.defineProperty(row, "clientWidth", {
      configurable: true,
      get: () => containerWidth,
    });

    const measure = row
      .closest('[data-testid="skill-triggers-test-wrap"]')
      ?.querySelector('[aria-hidden="true"]') as HTMLElement;
    Array.from(measure.children).forEach((child) => {
      Object.defineProperty(child, "offsetWidth", {
        configurable: true,
        get: () => pillWidth,
      });
    });

    act(() => {
      for (const cb of observedCallbacks) {
        cb([], {} as ResizeObserver);
      }
    });
  }

  const eventTriggerText = "pull_request_review_comment.created (github)";
  const modelText = "review-fast";

  const pills = [
    {
      id: "event-trigger",
      node: <span className={SKILL_CARD_PILL_CLASS}>{eventTriggerText}</span>,
    },
    {
      id: "model",
      node: <span className={SKILL_CARD_PILL_CLASS}>{modelText}</span>,
    },
  ];

  it("keeps pills on a single nowrap row with overflow handling", () => {
    renderWithProviders(
      <SkillCardPillRow testId="skill-triggers-test" pills={pills} />,
    );

    const row = screen.getByTestId("skill-triggers-test");
    expect(row).toHaveClass("flex-nowrap");
    expect(row).toHaveClass("overflow-hidden");
    expect(row).not.toHaveClass("flex-wrap");
  });

  it("folds pills that do not fit into a +N popover", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SkillCardPillRow testId="skill-triggers-test" pills={pills} />,
    );

    // Wide enough for one 80px pill + overflow reserve, not two.
    stubWidths(130, 80);

    await waitFor(() => {
      expect(
        screen.getByTestId("skill-triggers-test-overflow"),
      ).toBeInTheDocument();
    });

    expect(screen.getByTestId("skill-triggers-test")).toHaveTextContent(
      eventTriggerText,
    );
    expect(screen.getByTestId("skill-triggers-test")).not.toHaveTextContent(
      modelText,
    );

    const overflow = screen.getByTestId("skill-triggers-test-overflow");
    expect(overflow).toHaveAttribute(
      "aria-label",
      "SETTINGS$SKILLS_PILLS_OVERFLOW_ARIA",
    );

    await user.click(overflow);

    const popover = screen.getByTestId("skill-triggers-test-overflow-popover");
    expect(popover.parentElement).toBe(document.body);
    expect(
      within(popover).getByTestId("skill-triggers-test-overflow-item"),
    ).toHaveTextContent("review-fast");
  });

  it("opens the overflow popover without activating a wrapping card", async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();

    renderWithProviders(
      <div
        role="link"
        tabIndex={0}
        onClick={onActivate}
        onKeyDown={(event) => {
          if (event.key === "Enter") onActivate();
        }}
      >
        <SkillCardPillRow testId="skill-triggers-test" pills={pills} />
      </div>,
    );

    stubWidths(130, 80);

    await waitFor(() => {
      expect(
        screen.getByTestId("skill-triggers-test-overflow"),
      ).toBeInTheDocument();
    });

    await user.click(screen.getByTestId("skill-triggers-test-overflow"));

    expect(
      screen.getByTestId("skill-triggers-test-overflow-popover"),
    ).toBeInTheDocument();
    expect(onActivate).not.toHaveBeenCalled();
  });

  it("uses Escape for the overflow popover and keeps card activation scoped", async () => {
    const user = userEvent.setup();
    const onActivate = vi.fn();

    renderWithProviders(
      <div
        role="link"
        tabIndex={0}
        onClick={onActivate}
        onKeyDown={(event) => {
          if (event.key === "Enter" || event.key === " ") onActivate();
        }}
      >
        <SkillCardPillRow testId="skill-triggers-test" pills={pills} />
      </div>,
    );

    stubWidths(130, 80);
    await waitFor(() => {
      expect(
        screen.getByTestId("skill-triggers-test-overflow"),
      ).toBeInTheDocument();
    });

    const trigger = screen.getByTestId("skill-triggers-test-overflow");
    await user.click(trigger);
    await user.keyboard("{Escape}");

    expect(
      screen.queryByTestId("skill-triggers-test-overflow-popover"),
    ).not.toBeInTheDocument();
    expect(onActivate).not.toHaveBeenCalled();

    await user.keyboard("{Enter}");
    expect(
      screen.getByTestId("skill-triggers-test-overflow-popover"),
    ).toBeInTheDocument();
    expect(onActivate).not.toHaveBeenCalled();

    await user.keyboard("{Escape}");
    await user.keyboard(" ");
    expect(
      screen.getByTestId("skill-triggers-test-overflow-popover"),
    ).toBeInTheDocument();
    expect(onActivate).not.toHaveBeenCalled();
  });

  it("anchors the overflow popover below the +N pill", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <SkillCardPillRow testId="skill-triggers-test" pills={pills} />,
    );

    stubWidths(130, 80);

    await waitFor(() => {
      expect(
        screen.getByTestId("skill-triggers-test-overflow"),
      ).toBeInTheDocument();
    });

    const overflow = screen.getByTestId("skill-triggers-test-overflow");
    vi.spyOn(overflow, "getBoundingClientRect").mockReturnValue({
      x: 200,
      y: 100,
      top: 100,
      bottom: 118,
      left: 200,
      right: 236,
      width: 36,
      height: 18,
      toJSON: () => ({}),
    });

    await user.click(overflow);

    const popover = screen.getByTestId("skill-triggers-test-overflow-popover");
    expect(popover).toHaveStyle({
      position: "fixed",
      top: "122px",
      left: "200px",
    });
  });
});

describe("skill-facet-rail", () => {
  function buildGroups(): SkillFacetGroup[] {
    return [
      {
        id: "category",
        labelKey: I18nKey.SETTINGS$SKILLS_FACET_CATEGORY,
        rows: [
          {
            value: "environment",
            labelKey: I18nKey.SETTINGS$SKILLS_CATEGORY_ENVIRONMENT,
            count: 10,
            checked: false,
            disabled: false,
          },
          {
            value: "writing",
            labelKey: I18nKey.SETTINGS$SKILLS_CATEGORY_WRITING,
            count: 0,
            checked: false,
            disabled: true,
          },
        ],
      },
    ];
  }

  describe("SkillFacetRail", () => {
    it("renders nothing when no group is visible", () => {
      render(<SkillFacetRail groups={[]} onToggle={vi.fn()} />);
      expect(screen.queryByTestId("skill-facet-rail")).not.toBeInTheDocument();
    });

    it("renders each row with its count and checked state", () => {
      render(<SkillFacetRail groups={buildGroups()} onToggle={vi.fn()} />);

      const row = screen.getByTestId("skill-facet-category-environment");
      expect(row).toHaveAttribute("aria-checked", "false");
      expect(row).toHaveTextContent("10");
      expect(
        screen.getByTestId("skill-facet-group-category"),
      ).toBeInTheDocument();
    });

    it("reports the group and value when a row is clicked", async () => {
      const user = userEvent.setup();
      const onToggle = vi.fn();
      render(<SkillFacetRail groups={buildGroups()} onToggle={onToggle} />);

      await user.click(screen.getByTestId("skill-facet-category-environment"));

      expect(onToggle).toHaveBeenCalledWith("category", "environment");
    });

    it("does not report clicks on a disabled zero-count row", async () => {
      const user = userEvent.setup();
      const onToggle = vi.fn();
      render(<SkillFacetRail groups={buildGroups()} onToggle={onToggle} />);

      await user.click(screen.getByTestId("skill-facet-category-writing"));

      expect(onToggle).not.toHaveBeenCalled();
    });
  });
});

describe("skill-filter", () => {
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

  /**
   * The filter helpers take enablement as a predicate — resolving the allow-list
   * and deny-list behind it is the caller's job — so these tests express the
   * cases they care about as a deny-set.
   */
  function enabledExcept(...names: string[]) {
    const denied = new Set(names);
    return (skill: SkillInfo) => !denied.has(skill.name);
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
          "q=deno&source=personal&source=public&category=environment&type=repo&state=disabled&recommendation=recommended",
        ),
      );

      expect(state.query).toBe("deno");
      expect([...state.sources]).toEqual(["personal", "public"]);
      expect([...state.categories]).toEqual(["environment"]);
      expect([...state.types]).toEqual(["repo"]);
      expect([...state.states]).toEqual(["disabled"]);
      expect([...state.recommendations]).toEqual(["recommended"]);
    });

    it.each([
      ["source", "source=nonsense"],
      ["category", "category=code-hostig"],
      ["type", "type=bogus"],
      ["state", "state=maybe"],
      ["recommendation", "recommendation=maybe"],
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
      expect(
        toSkillFilterSearchParams(EMPTY_SKILL_FILTER_STATE).toString(),
      ).toBe("");
    });
  });

  describe("applySkillFilters", () => {
    const skills = buildMixedSkills();
    const isEnabled = enabledExcept("prd");

    it("ORs values within a group", () => {
      const result = applySkillFilters(
        skills,
        isEnabled,
        stateWith({ categories: new Set(["environment", "writing"]) }),
      );
      expect(result.map((s) => s.name)).toEqual(["deno", "prd"]);
    });

    it("ANDs across groups", () => {
      const result = applySkillFilters(
        skills,
        isEnabled,
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
        isEnabled,
        stateWith({ categories: new Set(["other"]) }),
      );
      expect(result.map((s) => s.name)).toEqual(["house-rules"]);
    });

    it("matches the query against name, description, and triggers", () => {
      expect(
        applySkillFilters(
          skills,
          isEnabled,
          stateWith({ query: "Requirements" }),
        ).map((s) => s.name),
      ).toEqual(["prd"]);
      expect(
        applySkillFilters(skills, isEnabled, stateWith({ query: "deno" })).map(
          (s) => s.name,
        ),
      ).toEqual(["deno"]);
    });

    it("returns everything when no facet is selected", () => {
      expect(
        applySkillFilters(skills, isEnabled, EMPTY_SKILL_FILTER_STATE),
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
        enabledExcept(),
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
        enabledExcept(),
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
        enabledExcept("prd"),
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
        enabledExcept("prd"),
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
        enabledExcept("prd"),
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
      const isEnabled = enabledExcept("prd");

      const narrowed = buildSkillFacetGroups(
        skills,
        isEnabled,
        stateWith({ categories: new Set(["environment"]) }),
      );

      // Only `deno` survives, leaving Source and Type one value each among the matches; visibility comes from the raw list, so both groups still render.
      expect(narrowed.map((g) => g.id)).toEqual(
        expect.arrayContaining(["source", "type"]),
      );
    });

    it("keeps group visibility stable while a search query narrows results", () => {
      const skills = buildMixedSkills();
      const isEnabled = enabledExcept("prd");
      const unfiltered = buildSkillFacetGroups(
        skills,
        isEnabled,
        EMPTY_SKILL_FILTER_STATE,
      );

      // "Requirements" matches only `prd`, so a query-filtered denominator would collapse every group to one value.
      const narrowed = buildSkillFacetGroups(
        skills,
        isEnabled,
        stateWith({ query: "Requirements" }),
      );

      expect(narrowed.map((g) => g.id)).toEqual(unfiltered.map((g) => g.id));
    });

    it("adds a recommendation group once the catalog flags one of the skills", () => {
      // `add-skill` carries `defaultEnabled` in the bundled catalog; `deno` does
      // not, which is what gives the group two discriminating values.
      const groups = buildSkillFacetGroups(
        [buildSkill({ name: "add-skill" }), buildSkill({ name: "deno" })],
        enabledExcept(),
        EMPTY_SKILL_FILTER_STATE,
      );

      const recommendation = groups.find((g) => g.id === "recommendation")!;
      expect(recommendation.labelKey).toBe(
        I18nKey.SETTINGS$SKILLS_FACET_RECOMMENDATION,
      );
      expect(recommendation.rows.map((r) => [r.value, r.count])).toEqual([
        ["recommended", 1],
        ["other", 1],
      ]);
    });

    it("filters down to the catalog's recommended skills", () => {
      const skills = [
        buildSkill({ name: "add-skill" }),
        buildSkill({ name: "deno" }),
      ];

      expect(
        applySkillFilters(
          skills,
          enabledExcept(),
          stateWith({ recommendations: new Set(["recommended"]) }),
        ).map((s) => s.name),
      ).toEqual(["add-skill"]);
    });

    it("disables a zero-count row unless it is checked", () => {
      const groups = buildSkillFacetGroups(
        buildMixedSkills(),
        enabledExcept("prd"),
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
        const collections = [
          "sources",
          "categories",
          "types",
          "states",
        ] as const;
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
});

describe("skill-filters-modal", () => {
  function buildGroups(): SkillFacetGroup[] {
    return [
      {
        id: "type",
        labelKey: I18nKey.SETTINGS$SKILLS_FACET_TYPE,
        rows: [
          {
            value: "repo",
            labelKey: I18nKey.SETTINGS$SKILLS_TYPE_REPO,
            count: 1,
            checked: false,
            disabled: false,
          },
          {
            value: "knowledge",
            labelKey: I18nKey.SETTINGS$SKILLS_TYPE_KNOWLEDGE,
            count: 2,
            checked: false,
            disabled: false,
          },
        ],
      },
    ];
  }

  describe("SkillFiltersModal", () => {
    it("toggles a facet through the same rail the desktop uses", async () => {
      const user = userEvent.setup();
      const onToggle = vi.fn();

      render(
        <SkillFiltersModal
          groups={buildGroups()}
          activeCount={0}
          onToggle={onToggle}
          onClearAll={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      await user.click(screen.getByTestId("skill-facet-type-repo"));

      expect(onToggle).toHaveBeenCalledWith("type", "repo");
    });

    it("hides the clear action when nothing is filtered", () => {
      render(
        <SkillFiltersModal
          groups={buildGroups()}
          activeCount={0}
          onToggle={vi.fn()}
          onClearAll={vi.fn()}
          onClose={vi.fn()}
        />,
      );

      expect(
        screen.queryByTestId("skill-filters-modal-clear"),
      ).not.toBeInTheDocument();
    });

    it("clears filters and closes on demand", async () => {
      const user = userEvent.setup();
      const onClearAll = vi.fn();
      const onClose = vi.fn();

      render(
        <SkillFiltersModal
          groups={buildGroups()}
          activeCount={2}
          onToggle={vi.fn()}
          onClearAll={onClearAll}
          onClose={onClose}
        />,
      );

      await user.click(screen.getByTestId("skill-filters-modal-clear"));
      expect(onClearAll).toHaveBeenCalled();

      await user.click(screen.getByTestId("skill-filters-modal-done"));
      expect(onClose).toHaveBeenCalled();
    });
  });
});
