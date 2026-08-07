import { describe, expect, it } from "vitest";
import { filterByName } from "#/components/features/cloudai/cloudai-shared";

describe("filterByName", () => {
  const items = [
    { id: "certplat", name: "CertPlat EventCo" },
    { id: "otel-proxy", name: "otel-proxy" },
    { id: "entity-logos", name: "Entity Logos" },
  ];

  it("returns all items when query is empty", () => {
    expect(filterByName(items, "", (item) => [item.name, item.id])).toEqual(
      items,
    );
    expect(filterByName(items, "   ", (item) => [item.name, item.id])).toEqual(
      items,
    );
  });

  it("filters by name case-insensitively", () => {
    expect(
      filterByName(items, "cert", (item) => [item.name, item.id]).map(
        (item) => item.id,
      ),
    ).toEqual(["certplat"]);
  });

  it("also matches id fields", () => {
    expect(
      filterByName(items, "entity-logos", (item) => [item.name, item.id]).map(
        (item) => item.name,
      ),
    ).toEqual(["Entity Logos"]);
  });
});
