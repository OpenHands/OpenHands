import { HooksClient } from "@openhands/typescript-client/clients";
import type { HookConfig } from "@openhands/typescript-client";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import { getAgentServerClientOptions } from "./agent-server-client-options";

// `loadHooks()` spends this twice (version probe, then the call) on the
// conversation-start critical path; the SDK's 60s default would stall a launch.
const HOOKS_LOAD_TIMEOUT_MS = 5000;

class HooksService {
  /**
   * Load workspace hooks from the agent-server by calling POST /api/hooks.
   * Returns the HookConfig if the workspace has hooks, or null.
   * Gracefully returns null when no usable local backend is available
   * (cloud backend, unseeded registry, or older agent-server).
   * `projectDir` must be the workspace root: the agent-server checks
   * `<project_dir>/.openhands/hooks.json` literally and never walks parents.
   */
  static async loadWorkspaceHooks(
    projectDir?: string,
  ): Promise<HookConfig | null> {
    // getEffectiveLocalBackend() returns null for cloud backends AND for the
    // NO_BACKEND sentinel (unseeded/empty registry), avoiding the silent
    // NoBackendAvailableError that getActiveBackend() would cause.
    if (!getEffectiveLocalBackend()) {
      return null;
    }

    try {
      return await HooksService.fetchWorkspaceHooks(
        projectDir ?? getAgentServerWorkingDir(),
      );
    } catch (error) {
      // Agent-server may not support the hooks endpoint or may be
      // unreachable; gracefully fall back to null.
      console.warn(
        "Failed to load workspace hooks, continuing without:",
        error,
      );
      return null;
    }
  }

  /**
   * POST /api/hooks for `projectDir` (a workspace root, as above) on the
   * active local backend. Unlike `loadWorkspaceHooks()` it rejects when the
   * agent-server is unreachable or lacks the endpoint, so a view can tell a
   * failure from a workspace without hooks (`null`).
   */
  static async fetchWorkspaceHooks(
    projectDir: string,
  ): Promise<HookConfig | null> {
    const response = await new HooksClient(
      getAgentServerClientOptions({ timeout: HOOKS_LOAD_TIMEOUT_MS }),
    ).loadHooks({ project_dir: projectDir });
    return response?.hook_config ?? null;
  }
}

export default HooksService;
