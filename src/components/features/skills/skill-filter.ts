import type { SkillCategoryId } from "@openhands/extensions/skills";
import { I18nKey } from "#/i18n/declaration";
import type { SkillInfo, SkillType } from "#/types/settings";
import {
  getSkillCategory,
  SKILL_CATEGORY_LABEL_KEYS,
  SKILL_CATEGORY_ORDER,
} from "#/utils/skill-category";
import {
  getSkillScope,
  SKILL_SCOPE_ORDER,
  type SkillScope,
} from "#/utils/skill-scope";
import { getSkillCardDescription } from "./get-skill-card-description";

export const SKILL_FILTER_QUERY_PARAM = "q";
const SOURCE_PARAM = "source";
const CATEGORY_PARAM = "category";
const TYPE_PARAM = "type";
const STATE_PARAM = "state";

export type SkillEnabledState = "enabled" | "disabled";
export type SkillFacetGroupId = "state" | "source" | "category" | "type";

const SKILL_ENABLED_STATE_ORDER: readonly SkillEnabledState[] = [
  "enabled",
  "disabled",
];
const SKILL_TYPE_ORDER: readonly SkillType[] = [
  "agentskills",
  "knowledge",
  "repo",
];

export interface SkillFilterState {
  query: string;
  sources: Set<SkillScope>;
  categories: Set<SkillCategoryId>;
  types: Set<SkillType>;
  states: Set<SkillEnabledState>;
}

export interface SkillFacetRowModel {
  value: string;
  labelKey: I18nKey;
  count: number;
  checked: boolean;
  disabled: boolean;
}

export interface SkillFacetGroup {
  id: SkillFacetGroupId;
  labelKey: I18nKey;
  rows: SkillFacetRowModel[];
}

export const EMPTY_SKILL_FILTER_STATE: SkillFilterState = {
  query: "",
  sources: new Set(),
  categories: new Set(),
  types: new Set(),
  states: new Set(),
};

/** Carrying `labelKey`s rather than translated strings keeps this module pure, so its tests need no i18n. */
interface GroupValue {
  value: string;
  labelKey: I18nKey;
}

function labelledValues<TValue extends string>(
  order: readonly TValue[],
  labelKeys: Record<TValue, I18nKey>,
): readonly GroupValue[] {
  return order.map((value) => ({ value, labelKey: labelKeys[value] }));
}

/** A new facet group is declared here and nowhere else: `selected` / `withSelected` are what keep every operation below driven by this table alone. */
interface GroupDef {
  id: SkillFacetGroupId;
  labelKey: I18nKey;
  param: string;
  values: readonly GroupValue[];
  valueOf: (skill: SkillInfo, disabledSet: Set<string>) => string;
  selected: (state: SkillFilterState) => Set<string>;
  withSelected: (
    state: SkillFilterState,
    next: Set<string>,
  ) => SkillFilterState;
}

/** Rebuilding rather than casting keeps every writer below cast-free: a value the group does not declare cannot survive. */
function narrowSet<TValue extends string>(
  order: readonly TValue[],
  next: Set<string>,
): Set<TValue> {
  return new Set(order.filter((value) => next.has(value)));
}

const SOURCE_LABEL_KEYS: Record<SkillScope, I18nKey> = {
  project: I18nKey.SETTINGS$SKILLS_SOURCE_PROJECT,
  personal: I18nKey.SETTINGS$SKILLS_SOURCE_PERSONAL,
  public: I18nKey.SETTINGS$SKILLS_SOURCE_PUBLIC,
};

const TYPE_LABEL_KEYS: Record<SkillType, I18nKey> = {
  agentskills: I18nKey.SETTINGS$SKILLS_TYPE_AGENTSKILLS,
  knowledge: I18nKey.SETTINGS$SKILLS_TYPE_KNOWLEDGE,
  repo: I18nKey.SETTINGS$SKILLS_TYPE_REPO,
};

const STATE_LABEL_KEYS: Record<SkillEnabledState, I18nKey> = {
  enabled: I18nKey.SETTINGS$SKILLS_ENABLED,
  disabled: I18nKey.SETTINGS$SKILLS_DISABLED,
};

const GROUP_DEFS: readonly GroupDef[] = [
  {
    id: "state",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_STATE,
    param: STATE_PARAM,
    values: labelledValues(SKILL_ENABLED_STATE_ORDER, STATE_LABEL_KEYS),
    valueOf: (skill, disabledSet) =>
      disabledSet.has(skill.name) ? "disabled" : "enabled",
    selected: (state) => state.states,
    withSelected: (state, next) => ({
      ...state,
      states: narrowSet(SKILL_ENABLED_STATE_ORDER, next),
    }),
  },
  {
    id: "source",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_SOURCE,
    param: SOURCE_PARAM,
    values: labelledValues(SKILL_SCOPE_ORDER, SOURCE_LABEL_KEYS),
    valueOf: (skill) => getSkillScope(skill),
    selected: (state) => state.sources,
    withSelected: (state, next) => ({
      ...state,
      sources: narrowSet(SKILL_SCOPE_ORDER, next),
    }),
  },
  {
    id: "category",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_CATEGORY,
    param: CATEGORY_PARAM,
    values: labelledValues(SKILL_CATEGORY_ORDER, SKILL_CATEGORY_LABEL_KEYS),
    valueOf: (skill) => getSkillCategory(skill),
    selected: (state) => state.categories,
    withSelected: (state, next) => ({
      ...state,
      categories: narrowSet(SKILL_CATEGORY_ORDER, next),
    }),
  },
  {
    id: "type",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_TYPE,
    param: TYPE_PARAM,
    values: labelledValues(SKILL_TYPE_ORDER, TYPE_LABEL_KEYS),
    valueOf: (skill) => skill.type,
    selected: (state) => state.types,
    withSelected: (state, next) => ({
      ...state,
      types: narrowSet(SKILL_TYPE_ORDER, next),
    }),
  },
];

function groupDef(id: SkillFacetGroupId): GroupDef {
  const def = GROUP_DEFS.find((candidate) => candidate.id === id);
  if (!def) throw new Error(`Unknown skill facet group: ${id}`);
  return def;
}

/** Unknown values are dropped, so a hand-edited URL cannot smuggle one in. */
function parseSet(raw: string[], def: GroupDef): Set<string> {
  const legal = new Set(def.values.map(({ value }) => value));
  return new Set(
    raw.map((value) => value.trim()).filter((value) => legal.has(value)),
  );
}

export function parseSkillFilterState(
  params: URLSearchParams,
): SkillFilterState {
  return GROUP_DEFS.reduce<SkillFilterState>(
    (state, def) =>
      def.withSelected(state, parseSet(params.getAll(def.param), def)),
    {
      ...EMPTY_SKILL_FILTER_STATE,
      query: params.get(SKILL_FILTER_QUERY_PARAM) ?? "",
    },
  );
}

export function toSkillFilterSearchParams(
  state: SkillFilterState,
): URLSearchParams {
  const params = new URLSearchParams();
  if (state.query) params.set(SKILL_FILTER_QUERY_PARAM, state.query);

  // Canonical order keeps URLs deterministic regardless of click order.
  for (const def of GROUP_DEFS) {
    const selected = def.selected(state);
    for (const { value } of def.values) {
      if (selected.has(value)) params.append(def.param, value);
    }
  }

  return params;
}

function matchesQuery(skill: SkillInfo, query: string): boolean {
  const trimmed = query.trim();
  if (!trimmed) return true;

  const haystacks = [
    skill.name,
    getSkillCardDescription(skill),
    skill.description ?? "",
    skill.content ?? "",
    skill.license ?? "",
    skill.compatibility ?? "",
    ...(skill.triggers ?? []),
    ...(skill.allowed_tools ?? []),
  ];

  const lowered = trimmed.toLowerCase();
  return haystacks.some((value) => value.toLowerCase().includes(lowered));
}

/**
 * `exclude` omits one group's own selection so its rows stay counted against the other groups alone.
 * A row therefore reads "this many skills carry this value, under your other filters" — the usual disjunctive facet count — and not "your result set will be this big": the group ORs its values together, so on a group that already has a selection the two differ.
 */
function matchesFacets(
  skill: SkillInfo,
  disabledSet: Set<string>,
  state: SkillFilterState,
  exclude?: SkillFacetGroupId,
): boolean {
  return GROUP_DEFS.every((def) => {
    if (def.id === exclude) return true;
    const selected = def.selected(state);
    if (selected.size === 0) return true;
    return selected.has(def.valueOf(skill, disabledSet));
  });
}

export function applySkillFilters(
  skills: SkillInfo[],
  disabledSet: Set<string>,
  state: SkillFilterState,
): SkillInfo[] {
  return skills.filter(
    (skill) =>
      matchesQuery(skill, state.query) &&
      matchesFacets(skill, disabledSet, state),
  );
}

function countByValue(
  skills: SkillInfo[],
  def: GroupDef,
  disabledSet: Set<string>,
): Record<string, number> {
  const counts: Record<string, number> = {};
  for (const skill of skills) {
    const value = def.valueOf(skill, disabledSet);
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function buildGroup(
  def: GroupDef,
  allSkills: SkillInfo[],
  searched: SkillInfo[],
  disabledSet: Set<string>,
  state: SkillFilterState,
): SkillFacetGroup | null {
  // Visibility and the row set come from the raw list so the rail's shape
  // depends only on what the user has, never on the active filters. If they
  // tracked filtered counts, narrowing could make a group vanish mid-click.
  const rawCounts = countByValue(allSkills, def, disabledSet);
  const discriminating = def.values.filter(
    ({ value }) => (rawCounts[value] ?? 0) > 0,
  );
  const selected = def.selected(state);
  // A group with only one (or zero) discriminating values is normally hidden,
  // but a selection can still be carried in from the URL for a value that has
  // no matches for this user. Hiding the group then would enforce the filter
  // invisibly, with no row left to explain or undo it.
  if (discriminating.length < 2 && selected.size === 0) return null;

  const visibleValues = def.values.filter(
    ({ value }) => (rawCounts[value] ?? 0) > 0 || selected.has(value),
  );

  const candidates = searched.filter((skill) =>
    matchesFacets(skill, disabledSet, state, def.id),
  );
  const counts = countByValue(candidates, def, disabledSet);

  return {
    id: def.id,
    labelKey: def.labelKey,
    rows: visibleValues.map(({ value, labelKey }) => {
      const count = counts[value] ?? 0;
      const checked = selected.has(value);
      return {
        value,
        labelKey,
        count,
        checked,
        disabled: count === 0 && !checked,
      };
    }),
  };
}

export function buildSkillFacetGroups(
  skills: SkillInfo[],
  disabledSet: Set<string>,
  state: SkillFilterState,
): SkillFacetGroup[] {
  const searched = skills.filter((skill) => matchesQuery(skill, state.query));

  return GROUP_DEFS.map((def) =>
    buildGroup(def, skills, searched, disabledSet, state),
  ).filter((group): group is SkillFacetGroup => group !== null);
}

export function toggleSkillFilterValue(
  state: SkillFilterState,
  groupId: SkillFacetGroupId,
  value: string,
): SkillFilterState {
  const def = groupDef(groupId);
  // `withSelected` would drop an unknown value anyway; returning the same
  // state object keeps a no-op toggle from re-rendering or rewriting the URL.
  if (!def.values.some((candidate) => candidate.value === value)) return state;

  const next = new Set(def.selected(state));
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }

  return def.withSelected(state, next);
}

export function clearSkillFilterFacets(
  state: SkillFilterState,
): SkillFilterState {
  return { ...EMPTY_SKILL_FILTER_STATE, query: state.query };
}

export function countActiveFilters(state: SkillFilterState): number {
  // The query is excluded: it is already visible in its own input, so counting
  // it would double-report in both the Filters badge and the summary row.
  return GROUP_DEFS.reduce((total, def) => total + def.selected(state).size, 0);
}
