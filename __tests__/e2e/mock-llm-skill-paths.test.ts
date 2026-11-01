// @vitest-environment node
import path from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  buildAgentServerEnv,
  buildSafeDevConfig,
} from "../../scripts/dev-safe.mjs";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("mock-LLM user skill paths", () => {
  it("writes npm fixtures where the isolated Agent Server loads user skills", async () => {
    vi.stubEnv("MOCK_LLM_USER_SKILLS_HOST_DIR", undefined);
    const { STATE_DIR, USER_SKILLS_DIR } =
      await import("../../tests/e2e/mock-llm/utils/skill-test-helpers");
    const config = buildSafeDevConfig(process.cwd(), {
      OH_CANVAS_SAFE_STATE_DIR: STATE_DIR,
      LOCAL_BACKEND_API_KEY: "test-session-key",
      OH_SECRET_KEY: "test-secret-key",
    });

    const serverEnv = buildAgentServerEnv(config, { env: {} });

    expect(USER_SKILLS_DIR).toBe(
      path.join(serverEnv.OH_PERSISTENCE_DIR, "skills"),
    );
  });

  it("keeps Docker fixture writes on the configured host mount", async () => {
    const mountPath = path.resolve(".tmp", "mock-llm-user-skills");
    vi.stubEnv("MOCK_LLM_USER_SKILLS_HOST_DIR", mountPath);

    const { USER_SKILLS_DIR } =
      await import("../../tests/e2e/mock-llm/utils/skill-test-helpers");

    expect(USER_SKILLS_DIR).toBe(mountPath);
  });
});
