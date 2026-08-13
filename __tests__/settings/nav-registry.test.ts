import { afterEach, describe, expect, it } from "vitest";
import type { ReactElement } from "react";
import {
  clearSettingsNavEntries,
  getRegisteredSettingsNavPaths,
  getSettingsNavEntries,
  registerSettingsNavEntry,
} from "#/settings/nav-registry";
import type { SettingsContext } from "#/settings/registry";

const ICON = null as unknown as ReactElement;
const LOCAL: SettingsContext = {
  backendKind: "local",
  orgId: null,
  featureFlags: undefined,
};
const CLOUD: SettingsContext = {
  backendKind: "cloud",
  orgId: "org-123",
  featureFlags: undefined,
};

const entry = (
  overrides: Partial<Parameters<typeof registerSettingsNavEntry>[0]>,
) =>
  registerSettingsNavEntry({
    id: "page.x",
    to: "/settings/x",
    order: 10,
    icon: ICON,
    text: "TEXT",
    subtitle: "SUBTITLE",
    ...overrides,
  });

describe("settings nav registry", () => {
  afterEach(() => {
    clearSettingsNavEntries();
  });

  it("sorts by order, then by id as a stable tiebreak", () => {
    entry({ id: "page.b", to: "/settings/b", order: 20 });
    entry({ id: "page.a", to: "/settings/a", order: 20 });
    entry({ id: "page.first", to: "/settings/first", order: 5 });

    expect(getSettingsNavEntries(LOCAL).map((e) => e.id)).toEqual([
      "page.first",
      "page.a",
      "page.b",
    ]);
  });

  it("is idempotent by id (re-registering replaces, does not duplicate)", () => {
    entry({ id: "page.dupe", to: "/settings/dupe", order: 10 });
    entry({ id: "page.dupe", to: "/settings/dupe", order: 99 });

    const entries = getSettingsNavEntries(LOCAL);
    expect(entries).toHaveLength(1);
    expect(entries[0].order).toBe(99);
  });

  it("applies the when predicate against the provided context", () => {
    entry({
      id: "page.cloud-only",
      to: "/settings/cloud",
      order: 10,
      when: (ctx) => ctx.backendKind === "cloud",
    });
    entry({ id: "page.always", to: "/settings/always", order: 20 });

    expect(getSettingsNavEntries(LOCAL).map((e) => e.id)).toEqual([
      "page.always",
    ]);
    expect(getSettingsNavEntries(CLOUD).map((e) => e.id)).toEqual([
      "page.cloud-only",
      "page.always",
    ]);
  });

  it("gates on feature flags read from the context", () => {
    entry({
      id: "page.llm",
      to: "/settings/llm",
      order: 10,
      when: (ctx) => !ctx.featureFlags?.hide_llm_settings,
    });

    const hidden: SettingsContext = {
      ...LOCAL,
      featureFlags: { hide_llm_settings: true, hide_users_page: false },
    };
    expect(getSettingsNavEntries(hidden)).toHaveLength(0);
    expect(getSettingsNavEntries(LOCAL)).toHaveLength(1);
  });

  it("treats a throwing when predicate as hidden", () => {
    entry({
      id: "page.throws",
      to: "/settings/throws",
      order: 10,
      when: () => {
        throw new Error("boom");
      },
    });

    expect(getSettingsNavEntries(LOCAL)).toHaveLength(0);
  });

  it("reports all registered paths regardless of visibility", () => {
    entry({ id: "page.visible", to: "/settings/visible", order: 10 });
    entry({
      id: "page.hidden",
      to: "/settings/hidden",
      order: 20,
      when: () => false,
    });

    expect(getSettingsNavEntries(LOCAL).map((e) => e.to)).toEqual([
      "/settings/visible",
    ]);
    expect(getRegisteredSettingsNavPaths().sort()).toEqual([
      "/settings/hidden",
      "/settings/visible",
    ]);
  });
});
