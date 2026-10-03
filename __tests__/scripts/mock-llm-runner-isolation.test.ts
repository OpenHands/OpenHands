// @vitest-environment node
import { describe, expect, it, beforeEach, afterEach, vi } from "vitest";
import { homedir, tmpdir } from "node:os";
import path from "node:path";
import {
  mkdtempSync,
  mkdirSync,
  writeFileSync,
  existsSync,
  rmSync,
} from "node:fs";
import globalTeardown from "../../tests/e2e/mock-llm/utils/global-teardown";

describe("mock-LLM runner isolation", () => {
  const originalEnv = { ...process.env };

  beforeEach(() => {
    vi.resetModules();
    process.env = { ...originalEnv };
  });

  afterEach(() => {
    process.env = originalEnv;
  });

  describe("user skills directory isolation", () => {
    it("never resolves to the real user's homedir/.openhands/skills", async () => {
      delete process.env.MOCK_LLM_USER_SKILLS_HOST_DIR;
      delete process.env.MOCK_LLM_TEST_HOME;
      delete process.env.OH_CANVAS_SAFE_STATE_DIR;

      const helpers =
        await import("../../tests/e2e/mock-llm/utils/skill-test-helpers");
      const realUserSkills = path.join(homedir(), ".openhands", "skills");

      expect(helpers.USER_SKILLS_DIR).not.toBe(realUserSkills);
      expect(helpers.USER_SKILLS_DIR).toContain(".tmp");
    });

    it("prefers MOCK_LLM_USER_SKILLS_HOST_DIR when set", async () => {
      const customPath = "/custom/user/skills";
      process.env.MOCK_LLM_USER_SKILLS_HOST_DIR = customPath;

      const helpers =
        await import("../../tests/e2e/mock-llm/utils/skill-test-helpers");
      expect(helpers.USER_SKILLS_DIR).toBe(path.resolve(customPath));
    });

    it("uses MOCK_LLM_TEST_HOME/.openhands/skills when test home is set", async () => {
      delete process.env.MOCK_LLM_USER_SKILLS_HOST_DIR;
      const testHome = "/tmp/test-home-dir";
      process.env.MOCK_LLM_TEST_HOME = testHome;

      const helpers =
        await import("../../tests/e2e/mock-llm/utils/skill-test-helpers");
      expect(helpers.USER_SKILLS_DIR).toBe(
        path.join(path.resolve(testHome), ".openhands", "skills"),
      );
    });

    it("derives from OH_CANVAS_SAFE_STATE_DIR parent when neither override is set", async () => {
      delete process.env.MOCK_LLM_USER_SKILLS_HOST_DIR;
      delete process.env.MOCK_LLM_TEST_HOME;
      const stateDir = "/tmp/mock-llm-123/state";
      process.env.OH_CANVAS_SAFE_STATE_DIR = stateDir;

      const helpers =
        await import("../../tests/e2e/mock-llm/utils/skill-test-helpers");
      expect(helpers.USER_SKILLS_DIR).toBe(
        path.join(
          path.dirname(path.resolve(stateDir)),
          "home",
          ".openhands",
          "skills",
        ),
      );
    });
  });

  describe("globalTeardown", () => {
    let tempDir: string;

    beforeEach(() => {
      tempDir = mkdtempSync(path.join(tmpdir(), "teardown-test-"));
    });

    afterEach(() => {
      if (existsSync(tempDir)) {
        rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it("removes MOCK_LLM_RUN_DIR during normal teardown", async () => {
      const dummyRunDir = path.join(tempDir, "mock-run");
      mkdirSync(dummyRunDir, { recursive: true });
      writeFileSync(path.join(dummyRunDir, "state.txt"), "data");
      expect(existsSync(dummyRunDir)).toBe(true);

      process.env.MOCK_LLM_RUN_DIR = dummyRunDir;
      delete process.env.MOCK_LLM_PRESERVE_STATE;

      await globalTeardown();
      expect(existsSync(dummyRunDir)).toBe(false);
    });

    it("preserves MOCK_LLM_RUN_DIR when MOCK_LLM_PRESERVE_STATE is truthy", async () => {
      const dummyRunDir = path.join(tempDir, "mock-run-preserved");
      mkdirSync(dummyRunDir, { recursive: true });
      writeFileSync(path.join(dummyRunDir, "state.txt"), "data");

      process.env.MOCK_LLM_RUN_DIR = dummyRunDir;
      process.env.MOCK_LLM_PRESERVE_STATE = "1";

      await globalTeardown();
      expect(existsSync(dummyRunDir)).toBe(true);
    });

    it("refuses to delete root, homedir, or cwd", async () => {
      const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => {});
      process.env.MOCK_LLM_RUN_DIR = homedir();
      delete process.env.MOCK_LLM_PRESERVE_STATE;

      await globalTeardown();
      expect(warnSpy).toHaveBeenCalledWith(
        expect.stringContaining("Refusing to delete unsafe per-run state directory"),
      );
      warnSpy.mockRestore();
    });
  });
});
