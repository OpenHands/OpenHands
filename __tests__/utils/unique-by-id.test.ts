import { describe, expect, it } from "vitest";
import { uniqueById } from "#/utils/unique-by-id";

describe("uniqueById", () => {
  it("keeps the first copy of each id in the original order", () => {
    // Arrange
    const items = [
      { id: "a", copy: 1 },
      { id: "b", copy: 1 },
      { id: "a", copy: 2 },
      { id: "c", copy: 1 },
      { id: "b", copy: 2 },
    ];

    // Act
    const result = uniqueById(items);

    // Assert
    expect(result).toEqual([
      { id: "a", copy: 1 },
      { id: "b", copy: 1 },
      { id: "c", copy: 1 },
    ]);
  });
});
