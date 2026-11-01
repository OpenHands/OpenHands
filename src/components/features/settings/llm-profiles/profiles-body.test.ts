import { describe, expect, it } from "vitest";
import { groupProfilesByConnection } from "./profiles-body";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import type { ProviderConnection } from "#/api/provider-connections-service/provider-connections-service.api";

const profile = (
  name: string,
  overrides: Partial<ProfileInfo> = {},
): ProfileInfo => ({
  name,
  model: "openai/gpt-4o",
  base_url: null,
  api_key_set: true,
  ...overrides,
});

const connection = (id: string): ProviderConnection => ({
  id,
  display_name: `Connection ${id}`,
  provider: "openai",
  base_url: null,
  created_at: 1,
  updated_at: 1,
  api_key_set: true,
});

describe("groupProfilesByConnection", () => {
  it("gives every connection a group in list order, including empty ones", () => {
    const groups = groupProfilesByConnection(
      [
        profile("b1", { provider_connection_id: "b" }),
        profile("a1", { provider_connection_id: "a" }),
        profile("b2", { provider_connection_id: "b" }),
      ],
      [connection("a"), connection("b"), connection("c")],
    );

    expect(
      groups.map((g) => [
        g.kind === "connection" ? g.connection.id : g.kind,
        g.profiles.map((p) => p.name),
      ]),
    ).toEqual([
      ["a", ["a1"]],
      ["b", ["b1", "b2"]],
      ["c", []],
    ]);
  });

  it("groups profiles linked to an unloaded connection under its id", () => {
    const groups = groupProfilesByConnection(
      [profile("x", { provider_connection_id: "missing" })],
      [],
    );

    expect(groups).toEqual([
      {
        kind: "unknown-connection",
        connectionId: "missing",
        profiles: [expect.objectContaining({ name: "x" })],
      },
    ]);
  });

  it("collects unlinked and broken-connection profiles into a trailing group", () => {
    const groups = groupProfilesByConnection(
      [
        profile("own-key"),
        profile("orphan", {
          provider_connection_id: "deleted",
          provider_connection_broken: true,
        }),
        profile("linked", { provider_connection_id: "a" }),
      ],
      [connection("a")],
    );

    expect(groups.map((g) => g.kind)).toEqual(["connection", "unlinked"]);
    expect(groups[1].profiles.map((p) => p.name)).toEqual([
      "own-key",
      "orphan",
    ]);
  });
});
