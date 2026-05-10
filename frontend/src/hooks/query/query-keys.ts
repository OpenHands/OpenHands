/**
 * Centralized query keys and cache configuration for TanStack Query.
 * Using constants ensures type safety and prevents typos.
 */

import { SettingsScope } from "#/types/settings";

export const QUERY_KEYS = {
  /** Web client configuration from the server */
  WEB_CLIENT_CONFIG: ["web-client-config"] as const,
} as const;

export const SETTINGS_QUERY_KEYS = {
  all: ["settings"] as const,
  byScope: (scope: SettingsScope, organizationId: string | null | undefined) =>
    ["settings", scope, organizationId] as const,
  personal: (organizationId: string | null | undefined) =>
    ["settings", "personal", organizationId] as const,
} as const;

/** Cache configuration shared across all config-related queries */
export const CONFIG_CACHE_OPTIONS = {
  staleTime: 1000 * 60 * 5, // 5 minutes
  gcTime: 1000 * 60 * 15, // 15 minutes
} as const;

export type QueryKeys = (typeof QUERY_KEYS)[keyof typeof QUERY_KEYS];
