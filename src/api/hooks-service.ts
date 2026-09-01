import { HooksClient } from "@openhands/typescript-client/clients";
import type { HookConfig } from "@openhands/typescript-client";
import { getAgentServerWorkingDir } from "./agent-server-config";
import { getEffectiveLocalBackend } from "./backend-registry/active-store";
import { getAgentServerClientOptions } from "./agent-server-client-options";

class HooksService {
  /**
   * Load workspace hooks from the agent-server by calling POST /api/hooks.
   * Returns the HookConfig if the workspace has hooks, or null.
   * Gracefully returns null when no usable local backend is available
   * (cloud backend, unseeded registry, or older agent-server).
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
      const response = await new HooksClient(
        getAgentServerClientOptions(),
      ).loadHooks({
        project_dir: projectDir ?? getAgentServerWorkingDir(),
      });
      return response?.hook_config ?? null;
    } catch {
      // Agent-server may not support the hooks endpoint or may be
      // unreachable; gracefully fall back to null.
      return null;
    }
  }
}

export default HooksService;
