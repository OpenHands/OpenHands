import { describe, it, expect } from "vitest";
import { buildProviderHostsFromConnections } from "#/hooks/use-provider-hosts";
import type { Provider } from "#/types/settings";

describe("buildProviderHostsFromConnections", () => {
  const connection = (
    providerType: Provider,
    host: string | null,
    connected = true,
  ) => ({
    providerType,
    connected,
    host,
  });

  it("builds mapping from single connection per provider", () => {
    const connections = [
      connection("gitlab", "gitlab.technogise.com"),
      connection("github", null),
    ];

    const result = buildProviderHostsFromConnections(connections);

    expect(result).toEqual({
      gitlab: "gitlab.technogise.com",
      github: null,
    });
  });

  it("uses first non-null host when multiple connections of same provider exist", () => {
    const connections = [
      connection("gitlab", null),
      connection("gitlab", "gitlab.technogise.com"),
      connection("gitlab", "other-gitlab.com"),
    ];

    const result = buildProviderHostsFromConnections(connections);

    // First connection sets gitlab to null, second upgrades to non-null
    expect(result).toEqual({
      gitlab: "gitlab.technogise.com",
    });
  });

  it("keeps first non-null host even if later connections have different hosts", () => {
    const connections = [
      connection("gitlab", "first-gitlab.com"),
      connection("gitlab", "second-gitlab.com"),
    ];

    const result = buildProviderHostsFromConnections(connections);

    // First non-null wins
    expect(result).toEqual({
      gitlab: "first-gitlab.com",
    });
  });

  it("ignores disconnected providers", () => {
    const connections = [
      connection("gitlab", "gitlab.technogise.com", false),
      connection("github", null, true),
    ];

    const result = buildProviderHostsFromConnections(connections);

    expect(result).toEqual({
      github: null,
    });
    expect(result).not.toHaveProperty("gitlab");
  });

  it("returns empty object for no connections", () => {
    const result = buildProviderHostsFromConnections([]);

    expect(result).toEqual({});
  });

  it("handles mixed providers with hosts and without", () => {
    const connections = [
      connection("github", null),
      connection("gitlab", "gitlab.example.com"),
      connection("bitbucket", null),
      connection("forgejo", "codeberg.org"),
    ];

    const result = buildProviderHostsFromConnections(connections);

    expect(result).toEqual({
      github: null,
      gitlab: "gitlab.example.com",
      bitbucket: null,
      forgejo: "codeberg.org",
    });
  });

  it("upgrades null to non-null when found later in the list", () => {
    const connections = [
      connection("gitlab", null),
      connection("gitlab", "gitlab.technogise.com"),
    ];

    const result = buildProviderHostsFromConnections(connections);

    expect(result).toEqual({
      gitlab: "gitlab.technogise.com",
    });
  });

  it("does not downgrade non-null host to null", () => {
    const connections = [
      connection("gitlab", "first-gitlab.com"),
      connection("gitlab", null),
    ];

    const result = buildProviderHostsFromConnections(connections);

    expect(result).toEqual({
      gitlab: "first-gitlab.com",
    });
  });
});
