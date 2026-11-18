/**
 * Property-based tests for self-hosted GitLab support.
 *
 * Tests the correctness properties defined in the spec:
 * 1. Host field normalization: non-empty → use value, else → null
 * 2. Provider-to-host mapping uses first non-null host
 * 3. Provider visibility matches providerHostKnown() result
 * 4. Self-hosted address format: https://{host}/{slug}
 * 5. Address building round-trip produces identical URL
 * 6. Invalid inputs return null
 * 7. Backward compatibility for hosted services
 */

import { describe, it, expect } from "vitest";
import * as fc from "fast-check";
import type { Provider } from "#/types/settings";
import { buildProviderHostsFromConnections } from "#/hooks/use-provider-hosts";
import {
  providerHostKnown,
  automationRepositoryAddress,
  providerBaseUrl,
} from "#/utils/automation-repo-sources";

// Valid provider types for testing
const PROVIDERS: Provider[] = [
  "github",
  "gitlab",
  "bitbucket",
  "bitbucket_data_center",
  "azure_devops",
  "forgejo",
];

// Providers that require a host to be known
const HOST_REQUIRED_PROVIDERS: Provider[] = [
  "gitlab",
  "forgejo",
  "bitbucket_data_center",
];

// Generators
const providerArb = fc.constantFrom(...PROVIDERS);

// Alphanumeric segment for hostnames (no special chars that could break URLs)
const hostnameSegmentArb = fc
  .stringMatching(/^[a-z][a-z0-9-]{0,20}[a-z0-9]$|^[a-z]$/)
  .filter((s) => s.length > 0 && !s.startsWith("-") && !s.endsWith("-"));

const validHostArb = fc.oneof(
  // Simple domain name
  fc
    .tuple(hostnameSegmentArb, fc.constantFrom("com", "org", "io", "dev", "local"))
    .map(([name, tld]) => `${name}.${tld}`),
  // Domain with subdomain
  fc
    .tuple(hostnameSegmentArb, hostnameSegmentArb, fc.constantFrom("com", "org", "io"))
    .map(([sub, name, tld]) => `${sub}.${name}.${tld}`),
  // With explicit https
  fc
    .tuple(hostnameSegmentArb, fc.constantFrom("com", "org"))
    .map(([name, tld]) => `https://${name}.${tld}`),
  // With port
  fc
    .tuple(hostnameSegmentArb, fc.constantFrom("com", "org"), fc.integer({ min: 1024, max: 65535 }))
    .map(([name, tld, port]) => `${name}.${tld}:${port}`),
  // Self-hosted examples
  fc.constantFrom(
    "gitlab.example.com",
    "git.company.internal",
    "gitlab.technogise.com",
    "https://git.example.com:8443",
    "code.mycompany.io",
  ),
);

const emptyHostArb = fc.constantFrom("", "   ", "\t", "\n", "  \t\n  ");

const nullishHostArb = fc.constantFrom(null, undefined);

// Alphanumeric repo slug (owner/repo format)
const repoSlugArb = fc
  .tuple(
    fc.stringMatching(/^[a-z][a-z0-9-]{0,20}$/).filter((s) => s.length > 0),
    fc.stringMatching(/^[a-z][a-z0-9-]{0,20}$/).filter((s) => s.length > 0),
  )
  .map(([owner, repo]) => `${owner}/${repo}`);

describe("Property 1: Host field normalization", () => {
  it("non-empty trimmed string is preserved", () => {
    fc.assert(
      fc.property(validHostArb, (host) => {
        const result = providerBaseUrl(host);
        // A valid host should produce a non-null base URL
        expect(result).not.toBeNull();
        // The result should be a valid URL
        expect(() => new URL(result!)).not.toThrow();
      }),
    );
  });

  it("empty or whitespace-only string normalizes to null base URL", () => {
    fc.assert(
      fc.property(emptyHostArb, (host) => {
        const result = providerBaseUrl(host);
        expect(result).toBeNull();
      }),
    );
  });

  it("null or undefined normalizes to null base URL", () => {
    fc.assert(
      fc.property(nullishHostArb, (host) => {
        const result = providerBaseUrl(host);
        expect(result).toBeNull();
      }),
    );
  });
});

describe("Property 2: First non-null host wins for duplicate providers", () => {
  it("first non-null host is used when multiple connections exist", () => {
    fc.assert(
      fc.property(
        providerArb,
        validHostArb,
        validHostArb,
        (provider, host1, host2) => {
          // Ensure hosts are different for meaningful test
          fc.pre(host1 !== host2);

          const connections = [
            { providerType: provider, connected: true, host: host1 },
            { providerType: provider, connected: true, host: host2 },
          ];

          const result = buildProviderHostsFromConnections(connections);

          // First non-null host should win
          expect(result[provider]).toBe(host1);
        },
      ),
    );
  });

  it("null host is upgraded to non-null if found later", () => {
    fc.assert(
      fc.property(providerArb, validHostArb, (provider, host) => {
        const connections = [
          { providerType: provider, connected: true, host: null },
          { providerType: provider, connected: true, host },
        ];

        const result = buildProviderHostsFromConnections(connections);

        // Non-null host should replace null
        expect(result[provider]).toBe(host);
      }),
    );
  });

  it("disconnected providers are not included in mapping", () => {
    fc.assert(
      fc.property(providerArb, validHostArb, (provider, host) => {
        const connections = [
          { providerType: provider, connected: false, host },
        ];

        const result = buildProviderHostsFromConnections(connections);

        // Disconnected provider should not be in the map
        expect(provider in result).toBe(false);
      }),
    );
  });

  it("multiple different providers maintain separate hosts", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(providerArb, validHostArb),
          { minLength: 1, maxLength: 6 },
        ),
        (providerHosts) => {
          const connections = providerHosts.map(([providerType, host]) => ({
            providerType,
            connected: true,
            host,
          }));

          const result = buildProviderHostsFromConnections(connections);

          // Each unique provider should have its first host
          const seen = new Map<Provider, string>();
          for (const [providerType, host] of providerHosts) {
            if (!seen.has(providerType)) {
              seen.set(providerType, host);
            }
          }

          for (const [providerType, expectedHost] of seen) {
            expect(result[providerType]).toBe(expectedHost);
          }
        },
      ),
    );
  });
});

describe("Property 3: Provider visibility matches providerHostKnown()", () => {
  it("host-required providers with valid host are known", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...HOST_REQUIRED_PROVIDERS),
        validHostArb,
        (provider, host) => {
          expect(providerHostKnown(provider, host)).toBe(true);
        },
      ),
    );
  });

  it("host-required providers without host are not known", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...HOST_REQUIRED_PROVIDERS),
        fc.oneof(nullishHostArb, emptyHostArb),
        (provider, host) => {
          expect(providerHostKnown(provider, host)).toBe(false);
        },
      ),
    );
  });

  it("github is always known regardless of host", () => {
    fc.assert(
      fc.property(
        fc.oneof(validHostArb, nullishHostArb, emptyHostArb),
        (host) => {
          expect(providerHostKnown("github", host)).toBe(true);
        },
      ),
    );
  });

  it("bitbucket (hosted) is known without host", () => {
    expect(providerHostKnown("bitbucket", null)).toBe(true);
    expect(providerHostKnown("bitbucket", undefined)).toBe(true);
  });

  it("azure_devops is known without host", () => {
    expect(providerHostKnown("azure_devops", null)).toBe(true);
    expect(providerHostKnown("azure_devops", undefined)).toBe(true);
  });
});

describe("Property 4: Self-hosted address format", () => {
  it("self-hosted GitLab produces https://{host}/{slug} format", () => {
    fc.assert(
      fc.property(validHostArb, repoSlugArb, (host, slug) => {
        const result = automationRepositoryAddress("gitlab", slug, host);
        expect(result).not.toBeNull();

        // Should contain the slug
        expect(result).toContain(slug);

        // Should be a valid URL
        expect(() => new URL(result!)).not.toThrow();

        // Should start with https
        expect(result).toMatch(/^https:\/\//);
      }),
    );
  });

  it("self-hosted Forgejo produces https://{host}/{slug} format", () => {
    fc.assert(
      fc.property(validHostArb, repoSlugArb, (host, slug) => {
        const result = automationRepositoryAddress("forgejo", slug, host);
        expect(result).not.toBeNull();
        expect(result).toContain(slug);
        expect(() => new URL(result!)).not.toThrow();
      }),
    );
  });

  it("Bitbucket Data Center produces /scm/ path format", () => {
    fc.assert(
      fc.property(validHostArb, repoSlugArb, (host, slug) => {
        const result = automationRepositoryAddress(
          "bitbucket_data_center",
          slug,
          host,
        );
        expect(result).not.toBeNull();
        expect(result).toContain("/scm/");
        expect(result).toMatch(/\.git$/);
      }),
    );
  });
});

describe("Property 5: Address building consistency", () => {
  it("building an address twice with same inputs produces identical result", () => {
    fc.assert(
      fc.property(
        providerArb,
        repoSlugArb,
        fc.oneof(validHostArb, fc.constant(null)),
        (provider, slug, host) => {
          const result1 = automationRepositoryAddress(provider, slug, host);
          const result2 = automationRepositoryAddress(provider, slug, host);
          expect(result1).toBe(result2);
        },
      ),
    );
  });

  it("address is deterministic for provider, slug, and host", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(providerArb, repoSlugArb, fc.oneof(validHostArb, fc.constant(null))),
          { minLength: 2, maxLength: 10 },
        ),
        (inputs) => {
          const results = new Map<string, string | null>();

          for (const [provider, slug, host] of inputs) {
            const key = `${provider}:${slug}:${host ?? "null"}`;
            const result = automationRepositoryAddress(provider, slug, host);

            if (results.has(key)) {
              expect(result).toBe(results.get(key));
            } else {
              results.set(key, result);
            }
          }
        },
      ),
    );
  });
});

describe("Property 6: Invalid inputs return null", () => {
  it("host-required provider without host returns null", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...HOST_REQUIRED_PROVIDERS),
        repoSlugArb,
        (provider, slug) => {
          const result = automationRepositoryAddress(provider, slug, null);
          expect(result).toBeNull();
        },
      ),
    );
  });

  it("host-required provider with empty host returns null", () => {
    fc.assert(
      fc.property(
        fc.constantFrom(...HOST_REQUIRED_PROVIDERS),
        repoSlugArb,
        emptyHostArb,
        (provider, slug, host) => {
          const result = automationRepositoryAddress(provider, slug, host);
          expect(result).toBeNull();
        },
      ),
    );
  });
});

describe("Property 7: Backward compatibility for hosted services", () => {
  it("GitHub without host returns shorthand (owner/repo)", () => {
    fc.assert(
      fc.property(repoSlugArb, (slug) => {
        const result = automationRepositoryAddress("github", slug, null);
        // GitHub without host should return the shorthand
        expect(result).toBe(slug);
      }),
    );
  });

  it("GitLab with gitlab.com host uses hosted URL", () => {
    fc.assert(
      fc.property(repoSlugArb, (slug) => {
        const result = automationRepositoryAddress("gitlab", slug, "gitlab.com");
        expect(result).toBe(`https://gitlab.com/${slug}`);
      }),
    );
  });

  it("Bitbucket without host uses hosted URL", () => {
    fc.assert(
      fc.property(repoSlugArb, (slug) => {
        const result = automationRepositoryAddress("bitbucket", slug, null);
        expect(result).toBe(`https://bitbucket.org/${slug}`);
      }),
    );
  });

  it("empty connections list produces empty mapping", () => {
    const result = buildProviderHostsFromConnections([]);
    expect(result).toEqual({});
  });

  it("all-disconnected connections produce empty mapping", () => {
    fc.assert(
      fc.property(
        fc.array(
          fc.tuple(providerArb, validHostArb),
          { minLength: 1, maxLength: 5 },
        ),
        (providerHosts) => {
          const connections = providerHosts.map(([providerType, host]) => ({
            providerType,
            connected: false,
            host,
          }));

          const result = buildProviderHostsFromConnections(connections);
          expect(result).toEqual({});
        },
      ),
    );
  });
});

describe("Edge cases", () => {
  it("handles hosts with various URL formats consistently", () => {
    const formats = [
      "gitlab.example.com",
      "https://gitlab.example.com",
      "https://gitlab.example.com/",
      "gitlab.example.com:8443",
      "https://gitlab.example.com:8443",
      "https://gitlab.example.com:8443/",
    ];

    for (const host of formats) {
      const result = providerBaseUrl(host);
      expect(result).not.toBeNull();
      // All formats should produce a valid URL
      expect(() => new URL(result!)).not.toThrow();
      // Should not have trailing slash
      expect(result).not.toMatch(/\/$/);
    }
  });

  it("preserves port numbers in hosts", () => {
    const result = providerBaseUrl("gitlab.example.com:8443");
    expect(result).toBe("https://gitlab.example.com:8443");
  });

  it("preserves path prefix in hosts", () => {
    const result = providerBaseUrl("https://example.com/gitlab");
    expect(result).toBe("https://example.com/gitlab");
  });
});
