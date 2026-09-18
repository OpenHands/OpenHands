import { describe, expect, it } from "vitest";

import type { Backend } from "#/api/backend-registry/types";
import {
  mergePortableBackends,
  parsePortableBackendConfig,
  serializePortableBackendConfig,
} from "#/api/backend-registry/portable-config";

const existingBackend: Backend = {
  id: "existing-id",
  name: "Office",
  host: "https://agents.example.com",
  apiKey: "old-key",
  kind: "local",
  connectionRevision: 4,
};

describe("portable backend configuration", () => {
  it("exports a versioned configuration without browser-local metadata", () => {
    const serialized = serializePortableBackendConfig([existingBackend]);

    expect(JSON.parse(serialized)).toEqual({
      version: 1,
      backends: [
        {
          name: "Office",
          url: "https://agents.example.com",
          sessionApiKey: "old-key",
          kind: "local",
        },
      ],
    });
  });

  it("rejects the whole file when any imported backend is malformed", () => {
    const raw = JSON.stringify({
      version: 1,
      backends: [
        {
          name: "Valid",
          url: "https://valid.example.com",
          sessionApiKey: "valid-key",
          kind: "local",
        },
        {
          name: "Invalid",
          url: "javascript:alert(1)",
          sessionApiKey: "invalid-key",
          kind: "local",
        },
      ],
    });

    expect(() => parsePortableBackendConfig(raw)).toThrow();
  });

  it("updates matching URLs in place and retains unrelated backends", () => {
    const retained: Backend = {
      id: "retained-id",
      name: "Retained",
      host: "http://localhost:8001",
      apiKey: "retained-key",
      kind: "local",
    };
    const imported = parsePortableBackendConfig(
      JSON.stringify({
        version: 1,
        backends: [
          {
            name: "Office Cloud",
            url: "https://AGENTS.example.com/",
            sessionApiKey: "new-key",
            kind: "cloud",
            authMode: "api-key",
          },
          {
            name: "Laptop",
            url: "http://localhost:9000",
            sessionApiKey: "laptop-key",
            kind: "local",
          },
        ],
      }),
    );

    const merged = mergePortableBackends(
      [existingBackend, retained],
      imported,
      () => "new-id",
    );

    expect(merged).toEqual([
      {
        id: "existing-id",
        name: "Office Cloud",
        host: "https://AGENTS.example.com",
        apiKey: "new-key",
        kind: "cloud",
        authMode: "api-key",
        connectionRevision: 5,
      },
      retained,
      {
        id: "new-id",
        name: "Laptop",
        host: "http://localhost:9000",
        apiKey: "laptop-key",
        kind: "local",
      },
    ]);
  });
});
