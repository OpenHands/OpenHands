import { describe, expect, it } from "vitest";
import type { Backend } from "#/api/backend-registry/types";
import {
  mergePortableBackends,
  parsePortableBackends,
  serializePortableBackends,
} from "#/utils/backend-config-portability";

const existing: Backend = {
  id: "local-a",
  name: "Local A",
  host: "http://localhost:8000",
  apiKey: "old-key",
  kind: "local",
};

describe("backend config portability", () => {
  it("round-trips connection data without browser-local ids", () => {
    const raw = serializePortableBackends([existing]);
    expect(raw).not.toContain('"id"');

    expect(parsePortableBackends(raw)).toEqual([
      {
        name: "Local A",
        host: "http://localhost:8000",
        apiKey: "old-key",
        kind: "local",
      },
    ]);
  });

  it("updates a matching URL in place and retains unrelated backends", () => {
    const other: Backend = {
      id: "cloud-b",
      name: "Cloud B",
      host: "https://cloud.example.com",
      apiKey: "cloud-key",
      kind: "cloud",
    };

    const merged = mergePortableBackends(
      [existing, other],
      [
        {
          name: "Renamed Local",
          host: "http://localhost:8000/",
          apiKey: "rotated-key",
          kind: "local",
        },
      ],
      () => "unused",
    );

    expect(merged).toHaveLength(2);
    expect(merged[0]).toMatchObject({
      id: "local-a",
      name: "Renamed Local",
      apiKey: "rotated-key",
      connectionRevision: 1,
    });
    expect(merged[1]).toEqual(other);
  });

  it("adds an imported backend that does not already exist", () => {
    const merged = mergePortableBackends(
      [existing],
      [
        {
          name: "Cloud",
          host: "https://cloud.example.com",
          apiKey: "key",
          kind: "cloud",
        },
      ],
      () => "imported-id",
    );

    expect(merged[1]).toMatchObject({
      id: "imported-id",
      host: "https://cloud.example.com",
    });
  });

  it("rejects malformed input without producing partial results", () => {
    expect(() => parsePortableBackends('{"version":1,"backends":[{"name":"x"}]}'))
      .toThrow("invalid entry");
    expect(() => parsePortableBackends("not-json")).toThrow("valid backend");
  });
});
