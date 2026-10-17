// @vitest-environment node
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  APPEARANCE_FILE,
  loadStoredAppearance,
  persistAppearance,
} from "../../../electron/lib/desktop-appearance.mjs";

let userDataPath: string;

beforeEach(() => {
  userDataPath = mkdtempSync(join(tmpdir(), "desktop-appearance-test-"));
});

afterEach(() => {
  rmSync(userDataPath, { recursive: true, force: true });
});

describe("desktop-appearance", () => {
  describe("loadStoredAppearance", () => {
    it("defaults to dark when no file exists", () => {
      expect(loadStoredAppearance(userDataPath)).toBe("dark");
    });

    it("defaults to dark when userDataPath is not provided", () => {
      expect(loadStoredAppearance(null)).toBe("dark");
      expect(loadStoredAppearance(undefined)).toBe("dark");
    });

    it("loads stored appearance when valid", () => {
      writeFileSync(
        join(userDataPath, APPEARANCE_FILE),
        JSON.stringify({ appearance: "light" }),
        "utf-8",
      );
      expect(loadStoredAppearance(userDataPath)).toBe("light");

      writeFileSync(
        join(userDataPath, APPEARANCE_FILE),
        JSON.stringify({ appearance: "dark" }),
        "utf-8",
      );
      expect(loadStoredAppearance(userDataPath)).toBe("dark");
    });

    it("defaults to dark when file is corrupted or contains unknown appearance", () => {
      writeFileSync(
        join(userDataPath, APPEARANCE_FILE),
        "invalid json content",
        "utf-8",
      );
      expect(loadStoredAppearance(userDataPath)).toBe("dark");

      writeFileSync(
        join(userDataPath, APPEARANCE_FILE),
        JSON.stringify({ appearance: "unknown" }),
        "utf-8",
      );
      expect(loadStoredAppearance(userDataPath)).toBe("dark");
    });
  });

  describe("persistAppearance", () => {
    it("writes appearance to file when valid", () => {
      persistAppearance(userDataPath, "light");
      const content = JSON.parse(
        readFileSync(join(userDataPath, APPEARANCE_FILE), "utf-8"),
      );
      expect(content).toEqual({ appearance: "light" });

      persistAppearance(userDataPath, "dark");
      const content2 = JSON.parse(
        readFileSync(join(userDataPath, APPEARANCE_FILE), "utf-8"),
      );
      expect(content2).toEqual({ appearance: "dark" });
    });

    it("ignores invalid appearance values or missing path", () => {
      persistAppearance(null, "light");
      persistAppearance(userDataPath, "invalid" as unknown as "dark" | "light");
      expect(loadStoredAppearance(userDataPath)).toBe("dark");
    });
  });
});
