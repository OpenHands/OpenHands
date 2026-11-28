import { describe, expect, it } from "vitest";
import { shouldPersistApiKeyInRecord } from "#/api/backend-registry/session-key-store";
import type { Backend } from "#/api/backend-registry/types";

function makeBackend(overrides: Partial<Backend> = {}): Backend {
  return {
    id: "backend-1",
    name: "Test Backend",
    host: "http://127.0.0.1:8000",
    apiKey: "",
    kind: "local",
    ...overrides,
  };
}

describe("shouldPersistApiKeyInRecord", () => {
  it("keeps loopback local session keys in the persisted record", () => {
    expect(shouldPersistApiKeyInRecord(makeBackend())).toBe(true);
    expect(
      shouldPersistApiKeyInRecord(
        makeBackend({ host: "http://localhost:18000" }),
      ),
    ).toBe(true);
  });

  it("keeps the public MARS guest key in the persisted record", () => {
    expect(
      shouldPersistApiKeyInRecord(
        makeBackend({
          host: "https://mars.example.com",
          apiKey: "mars-guest-key",
          marsSessionId: "session-1",
        }),
      ),
    ).toBe(true);
  });

  it("does not persist remote agent-server session keys in the record", () => {
    expect(
      shouldPersistApiKeyInRecord(
        makeBackend({
          host: "https://agent.example.com",
          apiKey: "remote-session-key",
        }),
      ),
    ).toBe(false);
  });

  it("treats an unparseable host as non-loopback", () => {
    expect(
      shouldPersistApiKeyInRecord(makeBackend({ host: "not-a-url" })),
    ).toBe(false);
  });

  it("does not persist cloud device-flow tokens in the record", () => {
    expect(
      shouldPersistApiKeyInRecord(
        makeBackend({
          host: "https://app.all-hands.dev",
          apiKey: "oauth-access-token",
          kind: "cloud",
          authMode: "api-key",
        }),
      ),
    ).toBe(false);
  });

  it("keeps cookie-auth cloud records, which carry no secret", () => {
    expect(
      shouldPersistApiKeyInRecord(
        makeBackend({
          host: "https://app.all-hands.dev",
          apiKey: "",
          kind: "cloud",
          authMode: "cookie",
        }),
      ),
    ).toBe(true);
  });
});
