import { describe, expect, it } from "vitest";
import {
  NEW_PROVIDER_CONNECTION,
  NO_PROVIDER_CONNECTION,
  findConnectionForProvider,
  resolveProviderConnectionSelection,
} from "./resolve-provider-connection";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";

const makeConnection = (
  overrides: Partial<ProviderConnection> = {},
): ProviderConnection => ({
  id: "conn-openai",
  display_name: "OpenAI",
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 1,
  api_key_set: true,
  ...overrides,
});

describe("findConnectionForProvider", () => {
  it("matches on the provider field, not the display name", () => {
    const renamed = makeConnection({ display_name: "Work key" });
    expect(findConnectionForProvider([renamed], "openai")).toBe(renamed);
  });

  it("returns undefined for an empty provider", () => {
    expect(findConnectionForProvider([makeConnection()], "")).toBeUndefined();
  });

  it("returns undefined when no connection matches", () => {
    expect(
      findConnectionForProvider([makeConnection()], "anthropic"),
    ).toBeUndefined();
  });
});

describe("resolveProviderConnectionSelection", () => {
  describe("no explicit choice (empty storedValue)", () => {
    it("reuses an existing connection for the model's provider", () => {
      const connection = makeConnection();
      const result = resolveProviderConnectionSelection({
        model: "openai/gpt-4o",
        storedValue: "",
        connections: [connection],
      });
      expect(result).toMatchObject({
        mode: "link",
        connectionId: "conn-openai",
        selectedKey: "conn-openai",
        provider: "openai",
        isOrphanedLink: false,
      });
    });

    it("defaults to create when the provider is known but has no connection", () => {
      const result = resolveProviderConnectionSelection({
        model: "anthropic/claude-3",
        storedValue: "",
        connections: [makeConnection()],
      });
      expect(result).toMatchObject({
        mode: "create",
        selectedKey: NEW_PROVIDER_CONNECTION,
        provider: "anthropic",
      });
    });

    it("falls back to inline when the model has no provider prefix", () => {
      const result = resolveProviderConnectionSelection({
        model: "gpt-4o",
        storedValue: "",
        connections: [],
      });
      expect(result).toMatchObject({
        mode: "none",
        selectedKey: NO_PROVIDER_CONNECTION,
        provider: "",
      });
    });
  });

  describe("explicit sentinels", () => {
    it("honors an explicit None even when a match exists", () => {
      const result = resolveProviderConnectionSelection({
        model: "openai/gpt-4o",
        storedValue: NO_PROVIDER_CONNECTION,
        connections: [makeConnection()],
      });
      expect(result.mode).toBe("none");
      expect(result.selectedKey).toBe(NO_PROVIDER_CONNECTION);
    });

    it("keeps create mode for an explicit New with a known provider", () => {
      const result = resolveProviderConnectionSelection({
        model: "openai/gpt-4o",
        storedValue: NEW_PROVIDER_CONNECTION,
        connections: [],
      });
      expect(result).toMatchObject({
        mode: "create",
        selectedKey: NEW_PROVIDER_CONNECTION,
        provider: "openai",
      });
    });

    it("degrades an explicit New to inline when the provider is unknown", () => {
      const result = resolveProviderConnectionSelection({
        model: "gpt-4o",
        storedValue: NEW_PROVIDER_CONNECTION,
        connections: [],
      });
      expect(result.mode).toBe("none");
      expect(result.selectedKey).toBe(NO_PROVIDER_CONNECTION);
    });
  });

  describe("explicit connection id", () => {
    it("links to an existing connection id", () => {
      const connection = makeConnection({ id: "conn-x", provider: "custom" });
      const result = resolveProviderConnectionSelection({
        model: "openai/gpt-4o",
        storedValue: "conn-x",
        connections: [connection],
      });
      expect(result).toMatchObject({
        mode: "link",
        connectionId: "conn-x",
        isOrphanedLink: false,
      });
    });

    it("flags an orphaned link when the id is absent from the list", () => {
      const result = resolveProviderConnectionSelection({
        model: "openai/gpt-4o",
        storedValue: "conn-gone",
        connections: [],
      });
      expect(result).toMatchObject({
        mode: "link",
        connectionId: "conn-gone",
        isOrphanedLink: true,
      });
    });
  });
});
