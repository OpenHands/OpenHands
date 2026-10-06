import { join } from "node:path";

/**
 * uv cache locations to keep when the mock-LLM stack overrides HOME.
 *
 * User skills are loaded from `$HOME/.openhands/skills`, so the stack sets
 * HOME to the isolated test home. uv resolves its cache and managed Pythons
 * from HOME as well (`$HOME/.cache/uv`, `$HOME/.local/share/uv/python`).
 * CI pre-warms those directories on the runner's real home; pointing uv at
 * the test home would miss that cache and reinstall under the webServer
 * timeout. Explicit UV_* overrides still win.
 */
export function uvCacheEnvForIsolatedHome(
  runnerHome: string,
  env: Record<string, string | undefined> = {},
): { UV_CACHE_DIR: string; UV_PYTHON_INSTALL_DIR: string } {
  const cacheOverride = env.UV_CACHE_DIR?.trim();
  const pythonOverride = env.UV_PYTHON_INSTALL_DIR?.trim();
  return {
    UV_CACHE_DIR: cacheOverride || join(runnerHome, ".cache", "uv"),
    UV_PYTHON_INSTALL_DIR:
      pythonOverride || join(runnerHome, ".local", "share", "uv", "python"),
  };
}
