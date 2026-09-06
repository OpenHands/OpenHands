import { describe, expect, it } from "vitest";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";
import {
  HOST_DETECTED_CONNECTION_PREFIX,
  isHostDetectedConnection,
  mergeHostDetectedConnections,
} from "#/utils/host-detected-provider-connections";

function connection(
  overrides: Partial<ProviderConnection> &
    Pick<ProviderConnection, "id" | "provider">,
): ProviderConnection {
  return {
    display_name: overrides.provider,
    base_url: null,
    created_at: 0,
    updated_at: 0,
    api_key_set: true,
    ...overrides,
  };
}

describe("mergeHostDetectedConnections", () => {
  it("prepends detected CLI logins when the stored list is empty", () => {
    const detected = [
      connection({
        id: `${HOST_DETECTED_CONNECTION_PREFIX}cursor-cli`,
        provider: "cursor-cli",
        display_name: "Cursor CLI",
      }),
    ];

    expect(mergeHostDetectedConnections([], detected)).toEqual(detected);
  });

  it("keeps a stored API-key connection next to a ChatGPT host login", () => {
    const stored = [
      connection({
        id: "conn-1",
        provider: "openai",
        display_name: "My OpenAI",
      }),
    ];
    const detected = [
      connection({
        id: `${HOST_DETECTED_CONNECTION_PREFIX}chatgpt`,
        provider: "openai",
        display_name: "ChatGPT subscription",
      }),
    ];

    expect(mergeHostDetectedConnections(stored, detected)).toEqual([
      detected[0],
      stored[0],
    ]);
  });

  it("does not duplicate a CLI that is already stored as a connection", () => {
    const stored = [
      connection({
        id: "conn-cursor",
        provider: "cursor-cli",
        display_name: "Cursor",
      }),
    ];
    const detected = [
      connection({
        id: `${HOST_DETECTED_CONNECTION_PREFIX}cursor-cli`,
        provider: "cursor-cli",
        display_name: "Cursor CLI",
      }),
    ];

    expect(mergeHostDetectedConnections(stored, detected)).toEqual(stored);
  });
});

describe("isHostDetectedConnection", () => {
  it("recognizes synthetic host rows", () => {
    expect(
      isHostDetectedConnection(
        connection({
          id: `${HOST_DETECTED_CONNECTION_PREFIX}opencode`,
          provider: "opencode",
        }),
      ),
    ).toBe(true);
    expect(
      isHostDetectedConnection(
        connection({ id: "conn-1", provider: "openai" }),
      ),
    ).toBe(false);
  });
});
