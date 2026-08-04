/**
 * Client for the skin management API mounted by scripts/static-server.mjs at
 * /skin-api (same origin as the served frontend, outside the /canvas base
 * path). A "skin" is a git-backed package that reconfigures this instance
 * and serves a custom web app that becomes the default tab of the UI.
 */

import { getAgentServerSessionApiKey } from "./agent-server-config";

export const SKIN_API_BASE = "/skin-api";
export const SKIN_APP_BASE = "/skin";

export interface SkinSecretDeclaration {
  name: string;
  description?: string;
}

/** Major theme colors a skin may declare in skin.yaml (`theme:`). */
export interface SkinTheme {
  accent?: string;
  background?: string;
  surface?: string;
  text?: string;
  muted?: string;
  border?: string;
  success?: string;
  warning?: string;
  danger?: string;
}

export interface SkinStatus {
  installed: boolean;
  running: boolean;
  name?: string;
  /** Lucide icon name (kebab-case, e.g. "activity") declared in skin.yaml,
   * shown on the skin's left-hand nav entry. Validated server-side; null
   * means "use the default skin icon". */
  icon?: string | null;
  screenshot?: string | null;
  repoUrl?: string;
  branch?: string | null;
  autoPush?: boolean;
  port?: number;
  canvasVersion?: string;
  canvasVersionRange?: string | null;
  secrets?: SkinSecretDeclaration[];
  theme?: SkinTheme | null;
  /**
   * Full set of CSS custom properties derived server-side from `theme`
   * (single source of truth in scripts/skin-service.mjs). Applied verbatim
   * to document.documentElement to retheme the whole Canvas UI.
   */
  themeVars?: Record<string, string> | null;
  error?: string | null;
}

export interface SkinPushResult {
  pushed: boolean;
  branch?: string;
  pullRequest?: { url: string; number: number } | null;
  error?: string;
}

async function skinFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const sessionApiKey = getAgentServerSessionApiKey();
  const response = await fetch(`${SKIN_API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(sessionApiKey ? { "X-Session-API-Key": sessionApiKey } : {}),
      ...(init.headers || {}),
    },
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(
      (body as { error?: string }).error || `HTTP ${response.status}`,
    );
  }
  return body as T;
}

export class SkinService {
  static getStatus(): Promise<SkinStatus> {
    return skinFetch<SkinStatus>("/status");
  }

  static install(params: {
    repoUrl: string;
    ref?: string;
    autoPush?: boolean;
  }): Promise<SkinStatus> {
    return skinFetch<SkinStatus>("/install", {
      method: "POST",
      body: JSON.stringify(params),
    });
  }

  static uninstall(): Promise<SkinStatus> {
    return skinFetch<SkinStatus>("/uninstall", { method: "POST" });
  }

  static pull(): Promise<SkinStatus> {
    return skinFetch<SkinStatus>("/pull", { method: "POST" });
  }

  static push(message?: string): Promise<SkinPushResult> {
    return skinFetch<SkinPushResult>("/push", {
      method: "POST",
      body: JSON.stringify({ message }),
    });
  }

  static exportConfiguration(): Promise<unknown> {
    return skinFetch<unknown>("/export", { method: "POST" });
  }

  static setAutoPush(autoPush: boolean): Promise<SkinStatus> {
    return skinFetch<SkinStatus>("/settings", {
      method: "PATCH",
      body: JSON.stringify({ autoPush }),
    });
  }
}
