import { describe, expect, it } from "vitest";
import {
  DEFAULT_ENABLED_SKILL_NAMES,
  SKILLS_CATALOG,
} from "@openhands/extensions/skills";
import {
  buildSkillEnablementFilter,
  findInvokedCatalogSkill,
  isCatalogSkill,
  isRecommendedSkill,
  migrateSkillEnablement,
  resolveEnabledCatalogSkills,
  toSkillEnablement,
} from "#/utils/skill-enablement";

// Two anchors from the bundled catalog: `add-skill` carries `defaultEnabled`,
// `add-javadoc` is the language-specific kind of skill OpenHands#16302 asked
// not to be opted into.
const RECOMMENDED = "add-skill";
const OPTIONAL = "add-javadoc";
const LOCAL = "house-rules";

describe("catalog membership", () => {
  it("recognises bundled skills and ignores locally authored ones", () => {
    expect(isCatalogSkill(RECOMMENDED)).toBe(true);
    expect(isCatalogSkill(OPTIONAL)).toBe(true);
    expect(isCatalogSkill(LOCAL)).toBe(false);
  });

  it("marks only the catalog's default-enabled entries as recommended", () => {
    expect(isRecommendedSkill(RECOMMENDED)).toBe(true);
    expect(isRecommendedSkill(OPTIONAL)).toBe(false);
    expect(DEFAULT_ENABLED_SKILL_NAMES.length).toBeLessThan(
      SKILLS_CATALOG.length,
    );
  });
});

describe("resolveEnabledCatalogSkills", () => {
  it("falls back to the curated default when nothing is persisted", () => {
    expect(resolveEnabledCatalogSkills({})).toEqual([
      ...DEFAULT_ENABLED_SKILL_NAMES,
    ]);
  });

  it("uses the persisted allow-list verbatim, empty list included", () => {
    expect(resolveEnabledCatalogSkills({ enabledSkills: [OPTIONAL] })).toEqual([
      OPTIONAL,
    ]);
    expect(resolveEnabledCatalogSkills({ enabledSkills: [] })).toEqual([]);
  });
});

describe("buildSkillEnablementFilter", () => {
  it("keeps user- and project-authored skills on unless denied", () => {
    expect(buildSkillEnablementFilter({ enabledSkills: [] })(LOCAL)).toBe(true);
    expect(buildSkillEnablementFilter({ disabledSkills: [LOCAL] })(LOCAL)).toBe(
      false,
    );
  });

  it("requires a catalog skill to be on the allow-list", () => {
    expect(buildSkillEnablementFilter({})(OPTIONAL)).toBe(false);
    expect(buildSkillEnablementFilter({})(RECOMMENDED)).toBe(true);
    expect(
      buildSkillEnablementFilter({ enabledSkills: [OPTIONAL] })(OPTIONAL),
    ).toBe(true);
  });

  it("lets an unmigrated deny-list veto a default-enabled skill", () => {
    // Until the migration runs, an existing "I turned this off" lives in the
    // deny-list alone; the allow-list fallback must not switch it back on.
    expect(
      buildSkillEnablementFilter({ disabledSkills: [RECOMMENDED] })(
        RECOMMENDED,
      ),
    ).toBe(false);
  });
});

describe("migrateSkillEnablement", () => {
  it("does nothing once an allow-list is persisted", () => {
    expect(migrateSkillEnablement({ enabledSkills: [] })).toBeUndefined();
    expect(
      migrateSkillEnablement({ enabledSkills: [RECOMMENDED] }),
    ).toBeUndefined();
  });

  it("preserves an existing workspace's all-on-minus-deny-list set", () => {
    const migrated = migrateSkillEnablement({ disabledSkills: [OPTIONAL] });

    expect(migrated?.enabled_skills).toHaveLength(SKILLS_CATALOG.length - 1);
    expect(migrated?.enabled_skills).not.toContain(OPTIONAL);
    expect(migrated?.enabled_skills).toContain(RECOMMENDED);
    // The catalog name has moved to the allow-list; a leftover deny entry
    // would veto the skill if the user switched it back on later.
    expect(migrated?.disabled_skills).toEqual([]);
  });

  it("keeps local deny entries, which the allow-list does not cover", () => {
    const migrated = migrateSkillEnablement({
      disabledSkills: [OPTIONAL, LOCAL],
    });

    expect(migrated?.disabled_skills).toEqual([LOCAL]);
  });

  it("gives a fresh workspace the curated default", () => {
    expect(migrateSkillEnablement({})?.enabled_skills).toEqual([
      ...DEFAULT_ENABLED_SKILL_NAMES,
    ]);
    // A deny-list naming only local skills says nothing about the catalog.
    expect(
      migrateSkillEnablement({ disabledSkills: [LOCAL] })?.enabled_skills,
    ).toEqual([...DEFAULT_ENABLED_SKILL_NAMES]);
  });

  it("persists the set even for a fresh workspace", () => {
    // Without an explicit write, a catalog addition marked `defaultEnabled`
    // would switch itself on in a workspace that never asked for it.
    expect(migrateSkillEnablement({})).toBeDefined();
  });
});

describe("toSkillEnablement", () => {
  it("keeps an absent allow-list undefined, the unmigrated sentinel", () => {
    expect(toSkillEnablement({ disabled_skills: [LOCAL] })).toEqual({
      enabledSkills: undefined,
      disabledSkills: [LOCAL],
    });
  });
});

describe("findInvokedCatalogSkill", () => {
  it("resolves a skill's own slash command, which is what automation cards send", () => {
    expect(findInvokedCatalogSkill("/standup-digest:setup")).toBe(
      "slack-standup-digest",
    );
    expect(findInvokedCatalogSkill("/codereview please look at src/")).toBe(
      "code-review",
    );
  });

  it("resolves `/<skill-name>`, which the Use skill button inserts", () => {
    expect(findInvokedCatalogSkill(`/${OPTIONAL} MyClass.java`)).toBe(OPTIONAL);
  });

  it("only counts the leading token", () => {
    // Matching a `/word` anywhere in prose would re-admit most of the catalog.
    expect(
      findInvokedCatalogSkill("what would /standup-digest:setup do?"),
    ).toBeUndefined();
  });

  it("ignores an empty, absent, or unknown command", () => {
    expect(findInvokedCatalogSkill(undefined)).toBeUndefined();
    expect(findInvokedCatalogSkill("   ")).toBeUndefined();
    expect(findInvokedCatalogSkill("just a message")).toBeUndefined();
    expect(findInvokedCatalogSkill("/not-a-real-skill")).toBeUndefined();
  });
});
