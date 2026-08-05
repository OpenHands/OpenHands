import { SettingsClient } from "@openhands/typescript-client/clients";
import type {
  MCPConfigPatch,
  MCPServer,
  MCPServerPatch,
} from "@openhands/typescript-client";
import { DEFAULT_SETTINGS } from "#/services/settings";
import { Settings, SettingsSchema, SettingsValue } from "#/types/settings";
import { stringRecord } from "#/utils/mcp-config";
import { getActiveBackend } from "../backend-registry/active-store";
import {
  fetchCloudConversationSettingsSchema,
  fetchCloudSettings,
  fetchCloudSettingsSchema,
  saveCloudSettings,
} from "../cloud/settings-service.api";
import { getAgentServerClientOptions } from "../agent-server-client-options";

/**
 * Fields the agent-server stores under `misc_settings.app_preferences` (see
 * SDK `openhands.sdk.settings.AppPreferences`). Mirrored here as a flat
 * partial of Settings so the rest of the frontend can keep treating them as
 * top-level keys (`settings.language`, `settings.disabled_skills`, …).
 */
export const APP_PREFERENCE_FIELDS = [
  "language",
  "user_consents_to_analytics",
  "enable_sound_notifications",
  "git_user_name",
  "git_user_email",
  "title_llm_profile",
  "disabled_skills",
] as const;

export type AppPreferenceField = (typeof APP_PREFERENCE_FIELDS)[number];

export type AppPreferences = Partial<Pick<Settings, AppPreferenceField>>;

/**
 * Container for frontend-owned settings the agent doesn't interpret.
 * Mirrors the SDK `MiscSettings` model introduced in agent-server 1.27.
 *
 * Single nested category today (`app_preferences`). Adding a future
 * category (e.g. `ui_preferences`) is a non-breaking change for the wire
 * shape — it just adds another optional sibling here.
 */
export interface MiscSettings {
  app_preferences?: AppPreferences;
}

/**
 * Response from GET /api/settings
 * Mirrors the SettingsResponse model in the agent server.
 */
export interface SettingsApiResponse {
  agent_settings: Record<string, SettingsValue>;
  conversation_settings: Record<string, SettingsValue>;
  llm_api_key_is_set: boolean;
  /**
   * Frontend-owned settings the agent does not interpret (currently just
   * `app_preferences`). Added in agent-server 1.27; earlier servers omit
   * the field entirely, in which case the frontend falls back to defaults.
   */
  misc_settings?: MiscSettings;
}

/**
 * Request payload for PATCH /api/settings.
 *
 * `misc_settings_diff` is deep-merged into the persisted `misc_settings`
 * block, matching the semantics of `agent_settings_diff` /
 * `conversation_settings_diff`. A partial diff like
 * `{ app_preferences: { language: "fr" } }` updates only `language` and
 * leaves every other `app_preferences` field alone. Lists
 * (`disabled_skills`) are replaced wholesale rather than merged.
 */
export interface SettingsUpdateRequest {
  agent_settings_diff?: Record<string, SettingsValue>;
  conversation_settings_diff?: Record<string, SettingsValue>;
  misc_settings_diff?: MiscSettings;
  // Permit additional keys so this stays assignable to the underlying
  // typescript-client SDK type, which uses `[key: string]: unknown` for
  // forward compatibility.
  [key: string]: unknown;
}

const APP_PREFERENCE_FIELD_SET: ReadonlySet<string> = new Set(
  APP_PREFERENCE_FIELDS,
);

const isAppPreferenceField = (key: string): key is AppPreferenceField =>
  APP_PREFERENCE_FIELD_SET.has(key);

/**
 * Split known app-preference keys out of a save payload so callers can
 * route them through `misc_settings_diff.app_preferences` (local) or as
 * flat top-level keys (cloud), while the remaining fields flow into the
 * `agent_settings_diff` / `conversation_settings_diff` branches.
 */
const extractAppPreferences = (
  input: Record<string, unknown>,
): { extracted: AppPreferences; rest: Record<string, unknown> } => {
  const extracted: AppPreferences = {};
  const rest: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(input)) {
    if (isAppPreferenceField(key)) {
      (extracted as Record<string, unknown>)[key] = value;
    } else {
      rest[key] = value;
    }
  }
  return { extracted, rest };
};

/**
 * Secret exposure mode for X-Expose-Secrets header.
 *
 * - undefined: Returns redacted secrets ("**********")
 * - "encrypted": Returns cipher-encrypted values (safe for frontend to round-trip)
 * - "plaintext": Returns raw secret values (backend use only!)
 */
export type ExposeSecretsMode = "encrypted" | "plaintext" | undefined;

const deepClone = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

const mergeRecords = (
  base: Record<string, SettingsValue> | null | undefined,
  next: Record<string, SettingsValue> | null | undefined,
) => ({ ...(base ?? {}), ...(next ?? {}) });

/**
 * Retry helper for API calls with exponential backoff.
 */
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3,
  baseDelayMs: number = 500,
): Promise<T> {
  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      return await fn();
    } catch (error) {
      if (attempt >= maxRetries - 1) {
        throw error;
      }

      const delay = baseDelayMs * 2 ** attempt;

      await new Promise<void>((resolve) => {
        setTimeout(resolve, delay);
      });
    }
  }

  throw new Error("Retry attempts exhausted");
}

/**
 * In-memory cache for settings to avoid repeated network calls.
 * The cache is invalidated on save operations.
 */
let settingsCache: {
  /** Settings with redacted secrets for display */
  redacted: SettingsApiResponse | null;
  /** Settings with encrypted secrets for conversation start */
  encrypted: SettingsApiResponse | null;
  /** Timestamp when the cache was last populated */
  timestamp: number;
} = {
  redacted: null,
  encrypted: null,
  timestamp: 0,
};

const CACHE_TTL_MS = 5 * 60 * 1000; // 5 minutes

const isCacheValid = () => Date.now() - settingsCache.timestamp < CACHE_TTL_MS;

const clearCache = () => {
  settingsCache = { redacted: null, encrypted: null, timestamp: 0 };
};

const isRecord = (value: unknown): value is Record<string, unknown> =>
  !!value && typeof value === "object" && !Array.isArray(value);

const basicAuthHeader = (username: string, password: string): string => {
  const token = btoa(`${username}:${password}`);
  return `Basic ${token}`;
};

const headersFromMcpAuth = (
  auth: Record<string, unknown>,
): Record<string, string> | null => {
  switch (auth.strategy) {
    case "none":
      return {};
    case "api_key": {
      if (typeof auth.value !== "string" || !auth.value) return null;
      const header =
        typeof auth.header_name === "string" && auth.header_name
          ? auth.header_name
          : "Authorization";
      const value =
        header === "Authorization" ? `Bearer ${auth.value}` : auth.value;
      return { [header]: value };
    }
    case "bearer":
      if (typeof auth.value !== "string" || !auth.value) return null;
      return { Authorization: `Bearer ${auth.value}` };
    case "basic":
      if (
        typeof auth.username !== "string" ||
        typeof auth.password !== "string"
      ) {
        return null;
      }
      return { Authorization: basicAuthHeader(auth.username, auth.password) };
    case "header":
      return stringRecord(auth.headers) ?? {};
    case "oauth2": {
      const tokens = isRecord(auth.state) ? auth.state.tokens : undefined;
      if (!isRecord(tokens) || typeof tokens.access_token !== "string") {
        return null;
      }
      return { Authorization: `Bearer ${tokens.access_token}` };
    }
    default:
      return null;
  }
};

/**
 * Convert SDK `auth` credentials to the cloud's header-only storage shape.
 *
 * The cloud settings endpoint applies `mcp_config` as a merge patch, so a
 * credential change must also tombstone the headers the previous credential
 * produced — otherwise stale secrets survive a strategy switch (e.g. an
 * api_key's custom header after moving to bearer). Stored headers that the
 * new credential still produces (via the patch's carried `headers` or the
 * converted auth) are left untouched; any other stored header is cleared.
 */
const cloudCompatibleMcpConfig = async (value: unknown): Promise<unknown> => {
  if (!isRecord(value)) return value;

  const hasWrapper = isRecord(value.mcpServers);
  const serverMap: Record<string, unknown> = hasWrapper
    ? (value.mcpServers as Record<string, unknown>)
    : value;

  const needsStoredCredential =
    Object.values(serverMap).some(
      (server) =>
        isRecord(server) && isRecord(server.auth) && server.auth !== null,
    ) && getActiveBackend().backend.kind === "cloud";

  const storedHeadersByServer = new Map<string, Record<string, string>>();
  if (needsStoredCredential) {
    try {
      const stored = await fetchCloudSettings();
      const storedMcp = isRecord(stored.mcp_config) ? stored.mcp_config : {};
      for (const [name, server] of Object.entries(storedMcp)) {
        if (isRecord(server)) {
          const headers = stringRecord(
            (server as Record<string, unknown>).headers,
          );
          if (headers) storedHeadersByServer.set(name, headers);
        }
      }
    } catch {
      // Fall back to the patch-only conversion; a transient fetch failure
      // must not block saving the user's explicit edits.
    }
  }

  const converted = Object.fromEntries(
    Object.entries(serverMap).map(([name, server]) => {
      if (!isRecord(server)) return [name, server];
      if (server.auth === null) {
        const nextServer: Record<string, unknown> = {
          ...server,
          headers: null,
        };
        delete nextServer.auth;
        return [name, nextServer];
      }
      if (!isRecord(server.auth)) return [name, server];

      const authHeaders = headersFromMcpAuth(server.auth);
      if (authHeaders === null) return [name, server];

      const nextServer = { ...server };
      const existingHeaders = stringRecord(server.headers) ?? {};
      const mergedHeaders: Record<string, string | null> = {
        ...existingHeaders,
        ...authHeaders,
      };

      const storedHeaders = storedHeadersByServer.get(name);
      if (storedHeaders) {
        for (const key of Object.keys(storedHeaders)) {
          if (!(key in mergedHeaders)) mergedHeaders[key] = null;
        }
      }

      delete nextServer.auth;
      if (Object.keys(mergedHeaders).length > 0) {
        nextServer.headers = mergedHeaders;
      } else {
        delete nextServer.headers;
      }
      return [name, nextServer];
    }),
  );

  return hasWrapper ? { ...value, mcpServers: converted } : converted;
};

/**
 * Transform API response into Settings object with derived fields.
 */
const transformApiResponse = (
  response: SettingsApiResponse,
): Partial<Settings> => {
  const agentSettings = response.agent_settings ?? {};
  const conversationSettings = response.conversation_settings ?? {};

  const partial: Partial<Settings> = {
    agent_settings: agentSettings,
    conversation_settings: conversationSettings,
    llm_api_key_set: response.llm_api_key_is_set,
  };

  // App-level user preferences come back nested under
  // `misc_settings.app_preferences` from the local agent-server (added in
  // 1.27, restructured into the `misc_settings` container in the follow-up
  // refactor). Hoist them onto the flat Settings shape the rest of the
  // frontend already speaks. Older servers omit the `misc_settings` field
  // entirely; the migration helper invoked from `getSettings` promotes any
  // leftover localStorage values to the server on first run, so the omitted
  // case just falls back to defaults.
  const prefs = response.misc_settings?.app_preferences;
  if (prefs) {
    for (const key of APP_PREFERENCE_FIELDS) {
      const value = prefs[key];
      if (value !== undefined) {
        (partial as Record<string, unknown>)[key] = value;
      }
    }
  }

  return partial;
};

/**
 * Sync derived settings fields from agent_settings and conversation_settings.
 * This ensures backward compatibility with code that reads top-level fields.
 */
const syncDerivedSettings = (settings: Partial<Settings>): Settings => {
  const agentSettings = mergeRecords(
    DEFAULT_SETTINGS.agent_settings ?? {},
    settings.agent_settings ?? {},
  );
  const conversationSettings = mergeRecords(
    DEFAULT_SETTINGS.conversation_settings ?? {},
    settings.conversation_settings ?? {},
  );

  const merged = {
    ...deepClone(DEFAULT_SETTINGS),
    ...settings,
    provider_tokens_set: {
      ...(DEFAULT_SETTINGS.provider_tokens_set ?? {}),
      ...(settings.provider_tokens_set ?? {}),
    },
    agent_settings: agentSettings,
    conversation_settings: conversationSettings,
  } as Settings;

  const llm = agentSettings.llm as Record<string, SettingsValue> | undefined;
  const condenser = agentSettings.condenser as
    | Record<string, SettingsValue>
    | undefined;

  if (typeof agentSettings.agent === "string") {
    merged.agent = agentSettings.agent;
  }
  if (typeof llm?.model === "string" && llm.model.length > 0) {
    merged.llm_model = llm.model;
  }
  if (typeof llm?.base_url === "string") {
    merged.llm_base_url = llm.base_url;
  }
  // Note: api_key may be redacted ("**********") when fetched without expose header
  // We don't sync it to top-level llm_api_key to avoid overwriting with redacted value
  if (typeof condenser?.enabled === "boolean") {
    merged.enable_default_condenser = condenser.enabled;
  }
  if (typeof condenser?.max_size === "number") {
    merged.condenser_max_size = condenser.max_size;
  }
  if (typeof conversationSettings.confirmation_mode === "boolean") {
    merged.confirmation_mode = conversationSettings.confirmation_mode;
  }
  if (
    typeof conversationSettings.security_analyzer === "string" ||
    conversationSettings.security_analyzer === null
  ) {
    merged.security_analyzer = conversationSettings.security_analyzer as
      | string
      | null;
  }
  if (typeof conversationSettings.max_iterations === "number") {
    merged.max_iterations = conversationSettings.max_iterations;
  }

  merged.search_api_key_set = !!merged.search_api_key;

  return merged;
};

class SettingsService {
  /**
   * Fetch settings from the agent server API with retry logic.
   *
   * @param exposeSecrets - Controls how secrets are returned:
   *   - undefined: Secrets are redacted ("**********") - safe for display
   *   - "encrypted": Secrets are cipher-encrypted - safe for round-trip to start conversation
   *   - "plaintext": Raw secrets - DO NOT USE from frontend
   */
  static async fetchSettingsFromApi(
    exposeSecrets?: ExposeSecretsMode,
  ): Promise<SettingsApiResponse> {
    return withRetry(() =>
      new SettingsClient(getAgentServerClientOptions()).getSettings({
        exposeSecrets,
      }),
    ) as Promise<SettingsApiResponse>;
  }

  /**
   * Get settings for display (secrets are redacted).
   * Uses in-memory cache for performance.
   */
  static async getSettings(): Promise<Settings> {
    // Cloud uses a different settings shape (flat top-level fields
    // including provider_tokens_set, llm_model, etc.). Branch out before
    // touching the local-only cache: cloud responses bypass the local
    // SettingsApiResponse shape and feed straight into syncDerivedSettings
    // so cloud-native fields like provider_tokens_set reach the GUI's
    // useUserProviders → useAppInstallations → useGitRepositories chain.
    if (getActiveBackend().backend.kind === "cloud") {
      try {
        const cloud = await withRetry(() => fetchCloudSettings());
        return syncDerivedSettings(cloud);
      } catch (error) {
        console.warn("Failed to fetch cloud settings, using defaults:", error);
        return syncDerivedSettings({});
      }
    }

    // Check cache first
    if (isCacheValid() && settingsCache.redacted) {
      return syncDerivedSettings(transformApiResponse(settingsCache.redacted));
    }

    try {
      const response = await this.fetchSettingsFromApi();
      settingsCache.redacted = response;
      settingsCache.timestamp = Date.now();
      return syncDerivedSettings(transformApiResponse(response));
    } catch (error) {
      // If API fails, return defaults
      console.warn("Failed to fetch settings from API, using defaults:", error);
      return syncDerivedSettings({});
    }
  }

  /**
   * Get settings with encrypted secrets for starting conversations.
   * The encrypted secrets can be passed to the start conversation API
   * with secrets_encrypted=true for server-side decryption.
   *
   * @throws Error if encrypted settings cannot be fetched - conversations
   *   should not start with broken/redacted credentials.
   */
  static async getSettingsForConversation(): Promise<{
    agentSettings: Record<string, SettingsValue>;
    conversationSettings: Record<string, SettingsValue>;
    secretsEncrypted: boolean;
  }> {
    // Check cache first
    if (isCacheValid() && settingsCache.encrypted) {
      return {
        agentSettings: settingsCache.encrypted.agent_settings,
        conversationSettings: settingsCache.encrypted.conversation_settings,
        secretsEncrypted: true,
      };
    }

    // Fetch encrypted settings - this MUST succeed for conversations to work.
    // Do not fall back to redacted settings as that would cause auth failures.
    const response = await this.fetchSettingsFromApi("encrypted");
    settingsCache.encrypted = response;
    if (!settingsCache.timestamp) {
      settingsCache.timestamp = Date.now();
    }
    return {
      agentSettings: response.agent_settings,
      conversationSettings: response.conversation_settings,
      secretsEncrypted: true,
    };
  }

  static async getSettingsSchema(): Promise<SettingsSchema> {
    if (getActiveBackend().backend.kind === "cloud") {
      return (await fetchCloudSettingsSchema()) as SettingsSchema;
    }
    return (await new SettingsClient(
      getAgentServerClientOptions(),
    ).getAgentSchema()) as SettingsSchema;
  }

  static async getConversationSettingsSchema(): Promise<SettingsSchema> {
    if (getActiveBackend().backend.kind === "cloud") {
      return (await fetchCloudConversationSettingsSchema()) as SettingsSchema;
    }
    return (await new SettingsClient(
      getAgentServerClientOptions(),
    ).getConversationSchema()) as SettingsSchema;
  }

  /**
   * Apply one name-keyed MCP merge patch in exactly one request. The server
   * owns the stored catalog and secret preservation; Canvas never rebuilds
   * the catalog from redacted display settings.
   */
  static async patchMcpConfig(patch: MCPConfigPatch): Promise<boolean> {
    if (getActiveBackend().backend.kind === "cloud") {
      const mcpConfig = await cloudCompatibleMcpConfig(patch);
      await withRetry(() =>
        saveCloudSettings({
          agent_settings_diff: { mcp_config: mcpConfig as SettingsValue },
        }),
      );
    } else {
      await withRetry(() =>
        new SettingsClient(getAgentServerClientOptions()).updateSettings({
          agent_settings_diff: { mcp_config: patch },
        }),
      );
    }
    clearCache();
    return true;
  }

  static async patchMcpServer(
    settingsKey: string,
    patch: MCPServerPatch,
  ): Promise<boolean> {
    if (getActiveBackend().backend.kind === "cloud") {
      return SettingsService.patchMcpConfig({ [settingsKey]: patch });
    }
    await withRetry(() =>
      new SettingsClient(getAgentServerClientOptions()).patchMcpServer(
        settingsKey,
        patch,
      ),
    );
    clearCache();
    return true;
  }

  static async createMcpServer(
    settingsKey: string,
    server: MCPServer,
  ): Promise<boolean> {
    if (getActiveBackend().backend.kind === "cloud") {
      return SettingsService.patchMcpConfig({ [settingsKey]: server });
    }
    await withRetry(() =>
      new SettingsClient(getAgentServerClientOptions()).createMcpServer(
        settingsKey,
        server,
      ),
    );
    clearCache();
    return true;
  }

  static async deleteMcpServer(settingsKey: string): Promise<boolean> {
    if (getActiveBackend().backend.kind === "cloud") {
      return SettingsService.patchMcpConfig({ [settingsKey]: null });
    }
    await withRetry(() =>
      new SettingsClient(getAgentServerClientOptions()).deleteMcpServer(
        settingsKey,
      ),
    );
    clearCache();
    return true;
  }

  /**
   * Save settings to the agent server API.
   * Uses PATCH for incremental updates.
   */
  static async saveSettings(
    settings: Partial<Settings> & Record<string, unknown>,
  ): Promise<boolean> {
    // Split app-level user-preference fields (language, git identity, sound
    // notifications, analytics consent, disabled_skills) off and route them
    // through `misc_settings_diff.app_preferences` (local) or as flat
    // top-level keys (cloud). The local agent-server stores them under
    // `PersistedSettings.misc_settings.app_preferences`; the cloud accepts
    // them as flat keys on `POST /api/v1/settings`.
    const { extracted: appPreferences, rest } = extractAppPreferences(
      settings as Record<string, unknown>,
    );
    const hasAppPreferences = Object.keys(appPreferences).length > 0;

    const payload: SettingsUpdateRequest = {};

    // Extract agent_settings_diff
    const agentSettingsDiff = rest.agent_settings_diff as
      | Record<string, SettingsValue>
      | undefined;
    if (agentSettingsDiff && Object.keys(agentSettingsDiff).length > 0) {
      payload.agent_settings_diff = agentSettingsDiff;
    }

    // Extract conversation_settings_diff
    const conversationSettingsDiff = rest.conversation_settings_diff as
      | Record<string, SettingsValue>
      | undefined;
    if (
      conversationSettingsDiff &&
      Object.keys(conversationSettingsDiff).length > 0
    ) {
      payload.conversation_settings_diff = conversationSettingsDiff;
    }

    if (hasAppPreferences) {
      payload.misc_settings_diff = { app_preferences: appPreferences };
    }

    const isCloud = getActiveBackend().backend.kind === "cloud";
    if (isCloud) {
      const hasCloudWork =
        !!payload.agent_settings_diff ||
        !!payload.conversation_settings_diff ||
        hasAppPreferences;
      if (!hasCloudWork) {
        return true;
      }
      // Build the cloud payload from the same diffs, but as a separate
      // object so undefined keys don't appear in the call (saveCloudSettings
      // is called from tests with an exact-shape assertion).
      const cloudPayload: Parameters<typeof saveCloudSettings>[0] = {};
      if (payload.agent_settings_diff) {
        cloudPayload.agent_settings_diff = { ...payload.agent_settings_diff };
        if ("mcp_config" in cloudPayload.agent_settings_diff) {
          cloudPayload.agent_settings_diff.mcp_config =
            (await cloudCompatibleMcpConfig(
              cloudPayload.agent_settings_diff.mcp_config,
            )) as SettingsValue;
        }
      }
      if (payload.conversation_settings_diff) {
        cloudPayload.conversation_settings_diff =
          payload.conversation_settings_diff;
      }
      if (hasAppPreferences) {
        // The cloud `POST /api/v1/settings` takes app-preference fields as
        // flat top-level keys (not under `app_preferences_diff`).
        // `saveCloudSettings` re-flattens them onto the request body.
        cloudPayload.app_preferences = appPreferences;
      }
      await withRetry(() => saveCloudSettings(cloudPayload));
    } else {
      // The local agent-server PATCH /api/settings requires at least one of
      // the three diff fields. Skip the request entirely if nothing changed.
      const hasLocalDiffs =
        !!payload.agent_settings_diff ||
        !!payload.conversation_settings_diff ||
        !!payload.misc_settings_diff;
      if (!hasLocalDiffs) {
        return true;
      }
      await withRetry(() =>
        new SettingsClient(getAgentServerClientOptions()).updateSettings(
          payload,
        ),
      );
    }

    // Invalidate cache after successful save
    clearCache();

    return true;
  }

  /**
   * Invalidate the settings cache.
   * Call this when settings may have changed externally.
   */
  static invalidateCache(): void {
    clearCache();
  }
}

export default SettingsService;
