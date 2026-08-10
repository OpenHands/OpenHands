/**
 * Centralized query keys and cache configuration for TanStack Query.
 * Using constants ensures type safety and prevents typos.
 */

import { SettingsScope } from "#/types/settings";

export const QUERY_KEYS = {
  /** Web client configuration from the server */
  WEB_CLIENT_CONFIG: ["web-client-config"] as const,
  /** Same-origin OpenHands app cookie authentication status */
  MAIN_APP_COOKIE_AUTH: ["main-app-cookie-auth"] as const,
} as const;

export const SETTINGS_QUERY_KEYS = {
  all: ["settings"] as const,
  byScope: (scope: SettingsScope) => ["settings", scope] as const,
  personal: () => ["settings", "personal"] as const,
} as const;

export const LLM_PROFILES_QUERY_KEYS = {
  all: ["llm-profiles"] as const,
} as const;

export const AGENT_PROFILES_QUERY_KEYS = {
  all: ["agent-profiles"] as const,
} as const;

/** Fail fast when older backends lack the profile endpoint. */
export const AGENT_PROFILES_RETRY_OPTIONS = {
  retry: false,
} as const;

export const LLM_SUBSCRIPTION_QUERY_KEYS = {
  all: ["llm-subscription"] as const,
  openaiStatus: ["llm-subscription", "openai", "status"] as const,
  openaiModels: ["llm-subscription", "openai", "models"] as const,
} as const;

export const LOCAL_WORKSPACES_QUERY_KEYS = {
  all: ["local-workspaces"] as const,
} as const;

export const PLUGINS_QUERY_KEYS = {
  /** Dynamic marketplace catalog (used by `use-plugins-marketplace`). */
  marketplace: ["plugins-marketplace"] as const,
  /** Installed plugins from the local agent-server. */
  installed: ["plugins-installed"] as const,
  /** Locally-discovered ambient plugins (used by `use-local-plugins`). */
  local: ["plugins-local"] as const,
} as const;

export const SETUP_QUERY_KEYS = {
  /** What the deployment supports. The same answer for every setup entry. */
  capabilities: () => ["setup-capabilities"] as const,
} as const;

export const APP_UPDATE_QUERY_KEYS = {
  /** Latest published @openhands/agent-canvas version (npm `latest` dist-tag). */
  latestVersion: ["agent-canvas-latest-version"] as const,
} as const;

export const APP_LOGIN_QUERY_KEYS = {
  status: ["app-login", "status"] as const,
  session: ["app-login", "session"] as const,
  users: ["app-login", "users"] as const,
} as const;

/** Pentest RBAC capabilities (PROJETOSIN-182). */
export const PENTEST_CAPABILITIES_QUERY_KEYS = {
  all: ["pentest-capabilities"] as const,
  me: ["pentest-capabilities", "me"] as const,
} as const;

/** Engagement Manager list for workspace creation (PROJETOSIN-183/185). */
export const PENTEST_ENGAGEMENTS_QUERY_KEYS = {
  all: ["pentest-engagements"] as const,
} as const;

/** Findings Service list / detail / stats (PROJETOSIN-188). */
export const FINDINGS_QUERY_KEYS = {
  all: ["findings"] as const,
  lists: ["findings", "list"] as const,
  list: (engagementId: string, filters: Record<string, unknown>) =>
    ["findings", "list", engagementId, filters] as const,
  details: ["findings", "detail"] as const,
  detail: (id: string) => ["findings", "detail", id] as const,
  stats: (engagementId: string) => ["findings", "stats", engagementId] as const,
} as const;

/** Mobile APK artifacts (PROJETOSIN-192). */
export const MOBILE_ARTIFACTS_QUERY_KEYS = {
  all: ["mobile-artifacts"] as const,
  lists: ["mobile-artifacts", "list"] as const,
  list: (engagementId: string) =>
    ["mobile-artifacts", "list", engagementId] as const,
} as const;

export const APPWRITE_QUERY_KEYS = {
  all: ["appwrite"] as const,
  databases: ["appwrite", "databases"] as const,
  collections: (databaseId: string) =>
    ["appwrite", "collections", databaseId] as const,
  documents: (databaseId: string, collectionId: string) =>
    ["appwrite", "documents", databaseId, collectionId] as const,
  attributes: (databaseId: string, collectionId: string) =>
    ["appwrite", "attributes", databaseId, collectionId] as const,
  functions: ["appwrite", "functions"] as const,
  executions: (functionId: string) =>
    ["appwrite", "executions", functionId] as const,
  functionVariables: (functionId: string) =>
    ["appwrite", "function-variables", functionId] as const,
  variables: ["appwrite", "variables"] as const,
  buckets: ["appwrite", "buckets"] as const,
  files: (bucketId: string) => ["appwrite", "files", bucketId] as const,
} as const;

/** Cache configuration shared across all config-related queries */
export const CONFIG_CACHE_OPTIONS = {
  staleTime: 1000 * 60 * 5, // 5 minutes
  gcTime: 1000 * 60 * 15, // 15 minutes
} as const;

export type QueryKeys = (typeof QUERY_KEYS)[keyof typeof QUERY_KEYS];
