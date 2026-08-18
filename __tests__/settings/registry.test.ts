import { afterEach, describe, expect, it } from "vitest";
import {
  clearSettingsSections,
  getSettingsSections,
  registerSettingsSection,
  type SettingsContext,
} from "#/settings/registry";

const Noop = () => null;
const LOCAL: SettingsContext = { backendKind: "local" };
const CLOUD: SettingsContext = { backendKind: "cloud" };

describe("settings registry", () => {
  afterEach(() => {
    clearSettingsSections();
  });

  it("returns only sections registered for the requested page", () => {
    registerSettingsSection({
      id: "app.one",
      page: "/settings/app",
      order: 10,
      Component: Noop,
    });
    registerSettingsSection({
      id: "other.one",
      page: "/settings/other",
      order: 10,
      Component: Noop,
    });

    const ids = getSettingsSections("/settings/app", LOCAL).map((s) => s.id);
    expect(ids).toEqual(["app.one"]);
  });

  it("sorts by order, then by id as a stable tiebreak", () => {
    registerSettingsSection({
      id: "app.b",
      page: "/settings/app",
      order: 20,
      Component: Noop,
    });
    registerSettingsSection({
      id: "app.a",
      page: "/settings/app",
      order: 20,
      Component: Noop,
    });
    registerSettingsSection({
      id: "app.first",
      page: "/settings/app",
      order: 5,
      Component: Noop,
    });

    const ids = getSettingsSections("/settings/app", LOCAL).map((s) => s.id);
    expect(ids).toEqual(["app.first", "app.a", "app.b"]);
  });

  it("is idempotent by id (re-registering replaces, does not duplicate)", () => {
    registerSettingsSection({
      id: "app.dupe",
      page: "/settings/app",
      order: 10,
      Component: Noop,
    });
    registerSettingsSection({
      id: "app.dupe",
      page: "/settings/app",
      order: 99,
      Component: Noop,
    });

    const sections = getSettingsSections("/settings/app", LOCAL);
    expect(sections).toHaveLength(1);
    expect(sections[0].order).toBe(99);
  });

  it("applies the when predicate against the provided context", () => {
    registerSettingsSection({
      id: "app.cloud-only",
      page: "/settings/app",
      order: 10,
      when: (ctx) => ctx.backendKind === "cloud",
      Component: Noop,
    });
    registerSettingsSection({
      id: "app.always",
      page: "/settings/app",
      order: 20,
      Component: Noop,
    });

    expect(
      getSettingsSections("/settings/app", LOCAL).map((s) => s.id),
    ).toEqual(["app.always"]);
    expect(
      getSettingsSections("/settings/app", CLOUD).map((s) => s.id),
    ).toEqual(["app.cloud-only", "app.always"]);
  });

  it("treats a throwing when predicate as hidden", () => {
    registerSettingsSection({
      id: "app.throws",
      page: "/settings/app",
      order: 10,
      when: () => {
        throw new Error("boom");
      },
      Component: Noop,
    });

    expect(getSettingsSections("/settings/app", LOCAL)).toHaveLength(0);
  });
});
