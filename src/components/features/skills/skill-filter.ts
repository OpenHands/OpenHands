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

const VALUE_SEPARATOR = ",";

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

/**
 * One group's identity: how to read its value off a skill, which values are
 * legal and in what order, and how to reach its slice of the filter state.
 *
 * `labelKey`s are carried rather than translated strings so this whole module
 * stays pure and its tests need no i18n.
 */
interface GroupDef {
  id: SkillFacetGroupId;
  labelKey: I18nKey;
  param: string;
  values: readonly string[];
  labelKeyByValue: Record<string, I18nKey>;
  valueOf: (skill: SkillInfo, disabledSet: Set<string>) => string;
  selected: (state: SkillFilterState) => Set<string>;
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
    values: SKILL_ENABLED_STATE_ORDER,
    labelKeyByValue: STATE_LABEL_KEYS,
    valueOf: (skill, disabledSet) =>
      disabledSet.has(skill.name) ? "disabled" : "enabled",
    selected: (state) => state.states,
  },
  {
    id: "source",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_SOURCE,
    param: SOURCE_PARAM,
    values: SKILL_SCOPE_ORDER,
    labelKeyByValue: SOURCE_LABEL_KEYS,
    valueOf: (skill) => getSkillScope(skill),
    selected: (state) => state.sources,
  },
  {
    id: "category",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_CATEGORY,
    param: CATEGORY_PARAM,
    values: SKILL_CATEGORY_ORDER,
    labelKeyByValue: SKILL_CATEGORY_LABEL_KEYS,
    valueOf: (skill) => getSkillCategory(skill),
    selected: (state) => state.categories,
  },
  {
    id: "type",
    labelKey: I18nKey.SETTINGS$SKILLS_FACET_TYPE,
    param: TYPE_PARAM,
    values: SKILL_TYPE_ORDER,
    labelKeyByValue: TYPE_LABEL_KEYS,
    valueOf: (skill) => skill.type,
    selected: (state) => state.types,
  },
];

function groupDef(id: SkillFacetGroupId): GroupDef {
  const def = GROUP_DEFS.find((candidate) => candidate.id === id);
  if (!def) throw new Error(`Unknown skill facet group: ${id}`);
  return def;
}

function parseSet<T extends string>(
  raw: string | null,
  allowed: readonly string[],
): Set<T> {
  if (!raw) return new Set();
  const legal = new Set(allowed);
  const parsed = raw
    .split(VALUE_SEPARATOR)
    .map((value) => value.trim())
    .filter((value) => legal.has(value));
  return new Set(parsed as T[]);
}

export function parseSkillFilterState(
  params: URLSearchParams,
): SkillFilterState {
  return {
    query: params.get(SKILL_FILTER_QUERY_PARAM) ?? "",
    sources: parseSet<SkillScope>(params.get(SOURCE_PARAM), SKILL_SCOPE_ORDER),
    categories: parseSet<SkillCategoryId>(
      params.get(CATEGORY_PARAM),
      SKILL_CATEGORY_ORDER,
    ),
    types: parseSet<SkillType>(params.get(TYPE_PARAM), SKILL_TYPE_ORDER),
    states: parseSet<SkillEnabledState>(
      params.get(STATE_PARAM),
      SKILL_ENABLED_STATE_ORDER,
    ),
  };
}

export function toSkillFilterSearchParams(
  state: SkillFilterState,
): URLSearchParams {
  const params = new URLSearchParams();
  if (state.query) params.set(SKILL_FILTER_QUERY_PARAM, state.query);

  // Canonical order keeps URLs deterministic regardless of click order.
  for (const def of GROUP_DEFS) {
    const selected = def.selected(state);
    if (selected.size === 0) continue;
    params.set(
      def.param,
      def.values.filter((value) => selected.has(value)).join(VALUE_SEPARATOR),
    );
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
 * `exclude` omits one group's own selection, which is what makes each facet
 * row's count predict the result of clicking it.
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
    (value) => (rawCounts[value] ?? 0) > 0,
  );
  if (discriminating.length < 2) return null;

  const selected = def.selected(state);
  const visibleValues = def.values.filter(
    (value) => (rawCounts[value] ?? 0) > 0 || selected.has(value),
  );

  const candidates = searched.filter((skill) =>
    matchesFacets(skill, disabledSet, state, def.id),
  );
  const counts = countByValue(candidates, def, disabledSet);

  return {
    id: def.id,
    labelKey: def.labelKey,
    rows: visibleValues.map((value) => {
      const count = counts[value] ?? 0;
      const checked = selected.has(value);
      return {
        value,
        labelKey: def.labelKeyByValue[value]!,
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
  if (!def.values.includes(value)) return state;

  const next = new Set(def.selected(state));
  if (next.has(value)) {
    next.delete(value);
  } else {
    next.add(value);
  }

  switch (groupId) {
    case "state":
      return { ...state, states: next as Set<SkillEnabledState> };
    case "source":
      return { ...state, sources: next as Set<SkillScope> };
    case "category":
      return { ...state, categories: next as Set<SkillCategoryId> };
    case "type":
      return { ...state, types: next as Set<SkillType> };
    default:
      return state;
  }
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
