import { rmSync, existsSync } from "node:fs";

/**
 * Global teardown for mock-LLM E2E tests.
 * Removes the isolated per-run state and fixture directory after all tests finish,
 * unless MOCK_LLM_PRESERVE_STATE is explicitly set.
 */
export default async function globalTeardown(): Promise<void> {
  const runDir = process.env.MOCK_LLM_RUN_DIR;
  if (runDir && !process.env.MOCK_LLM_PRESERVE_STATE && existsSync(runDir)) {
    try {
      rmSync(runDir, { recursive: true, force: true });
    } catch (err) {
      console.warn(`Failed to clean per-run state directory ${runDir}:`, err);
    }
  }
}
