import { describe, expect, it } from "vitest";
import type { ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { groupProfilesByConnection } from "./profiles-body";

function profile(name: string, connectionId?: string | null): ProfileInfo {
  return {
    name,
    model: "openai/gpt-4o",
    base_url: null,
    api_key_set: true,
    provider_connection_id: connectionId ?? null,
  };
}

describe("groupProfilesByConnection", () => {
  it("groups linked profiles by connection, preserving order", () => {
    const groups = groupProfilesByConnection(
      [profile("a", "conn-1"), profile("b", "conn-2"), profile("c", "conn-1")],
      { "conn-1": "OpenAI", "conn-2": "Anthropic" },
    );

    expect(groups).toEqual([
      {
        connectionId: "conn-1",
        label: "OpenAI",
        profiles: [profile("a", "conn-1"), profile("c", "conn-1")],
      },
      {
        connectionId: "conn-2",
        label: "Anthropic",
        profiles: [profile("b", "conn-2")],
      },
    ]);
  });

  it("collects unlinked profiles into a trailing null group", () => {
    const groups = groupProfilesByConnection(
      [profile("a", "conn-1"), profile("b"), profile("c", null)],
      { "conn-1": "OpenAI" },
    );

    expect(groups.map((g) => g.connectionId)).toEqual(["conn-1", null]);
    expect(groups[1].label).toBeNull();
    expect(groups[1].profiles.map((p) => p.name)).toEqual(["b", "c"]);
  });

  it("falls back to the connection id when no display name is known", () => {
    const groups = groupProfilesByConnection([profile("a", "conn-x")], {});

    expect(groups[0].label).toBe("conn-x");
  });

  it("omits the unlinked group when every profile is linked", () => {
    const groups = groupProfilesByConnection([profile("a", "conn-1")], {
      "conn-1": "OpenAI",
    });

    expect(groups).toHaveLength(1);
    expect(groups[0].connectionId).toBe("conn-1");
  });
});
