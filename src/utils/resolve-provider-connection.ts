import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import { extractModelAndProvider } from "#/utils/extract-model-and-provider";

/**
 * Dropdown sentinel for "no provider connection" — the profile keeps its own
 * inline api_key / base_url (today's behavior).
 */
export const NO_PROVIDER_CONNECTION = "__none__";

/**
 * Dropdown sentinel for "create a new provider connection for this model's
 * provider on save". The actual connection is created lazily in the save flow
 * (see LlmSettingsLocalView), so nothing is persisted until the profile is
 * saved.
 */
export const NEW_PROVIDER_CONNECTION = "__new__";

/**
 * What the profile should do with credentials, derived from the current model,
 * the connection the user explicitly picked (if any), and the connections that
 * exist. This is a pure function so it can drive both the editor UI (which
 * option is selected, whether to show the inline key field) and the save flow
 * (link an existing connection, create one, or keep inline creds) from a single
 * source of truth.
 */
export type ProviderConnectionMode = "none" | "link" | "create";

export interface ProviderConnectionSelection {
  /** The dropdown key to render as selected. */
  selectedKey: string;
  /** What the save flow should do. */
  mode: ProviderConnectionMode;
  /** For `mode === "link"`: the connection id to reference. */
  connectionId?: string;
  /**
   * The litellm provider derived from the model (e.g. "openai"). Empty when the
   * model has no provider prefix. Used to name an auto-created connection and to
   * find an existing connection for the same provider.
   */
  provider: string;
  /**
   * True when `selectedKey` points at a connection id that is not in the
   * supplied list (its connection was deleted, or the list is still loading).
   * The caller surfaces this as its own dropdown option so the link can be seen
   * and cleared.
   */
  isOrphanedLink: boolean;
}

const deriveProvider = (model: string): string =>
  extractModelAndProvider(model).provider || "";

/**
 * Find an existing connection for a provider. Matching is on the connection's
 * `provider` field (the stable identity) rather than its display name, so a
 * user renaming a connection in the manager does not break auto-selection.
 */
export function findConnectionForProvider(
  connections: ProviderConnection[],
  provider: string,
): ProviderConnection | undefined {
  if (!provider) return undefined;
  return connections.find((connection) => connection.provider === provider);
}

/**
 * Resolve which provider-connection option a profile should use.
 *
 * `storedValue` is the explicit choice held in form state:
 * - a real connection id → link to it;
 * - {@link NO_PROVIDER_CONNECTION} → inline creds (user opted out);
 * - {@link NEW_PROVIDER_CONNECTION} → create-on-save for the model's provider;
 * - empty string → no explicit choice yet, so fall back to the default:
 *   reuse an existing connection for the model's provider, otherwise offer to
 *   create one, otherwise inline.
 */
export function resolveProviderConnectionSelection({
  model,
  storedValue,
  connections,
}: {
  model: string;
  storedValue: string;
  connections: ProviderConnection[];
}): ProviderConnectionSelection {
  const provider = deriveProvider(model);

  if (storedValue === NO_PROVIDER_CONNECTION) {
    return {
      selectedKey: NO_PROVIDER_CONNECTION,
      mode: "none",
      provider,
      isOrphanedLink: false,
    };
  }

  if (storedValue === NEW_PROVIDER_CONNECTION) {
    // Creating a new connection only makes sense with a known provider.
    if (!provider) {
      return {
        selectedKey: NO_PROVIDER_CONNECTION,
        mode: "none",
        provider,
        isOrphanedLink: false,
      };
    }
    return {
      selectedKey: NEW_PROVIDER_CONNECTION,
      mode: "create",
      provider,
      isOrphanedLink: false,
    };
  }

  if (storedValue) {
    // An explicit connection id.
    const exists = connections.some((c) => c.id === storedValue);
    return {
      selectedKey: storedValue,
      mode: "link",
      connectionId: storedValue,
      provider,
      isOrphanedLink: !exists,
    };
  }

  // No explicit choice: prefer reusing an existing connection for the provider.
  const match = findConnectionForProvider(connections, provider);
  if (match) {
    return {
      selectedKey: match.id,
      mode: "link",
      connectionId: match.id,
      provider,
      isOrphanedLink: false,
    };
  }

  // No match: offer to create one when the provider is known, else inline.
  if (provider) {
    return {
      selectedKey: NEW_PROVIDER_CONNECTION,
      mode: "create",
      provider,
      isOrphanedLink: false,
    };
  }

  return {
    selectedKey: NO_PROVIDER_CONNECTION,
    mode: "none",
    provider,
    isOrphanedLink: false,
  };
}
