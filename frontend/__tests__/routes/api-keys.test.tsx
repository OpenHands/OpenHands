import { describe, expect, it } from "vitest";
import { clientLoader } from "#/routes/api-keys";

describe("clientLoader permission checks", () => {
  it("should export a clientLoader for route protection", () => {
    // This test verifies the clientLoader is exported (for consistency with other routes)
    expect(clientLoader).toBeDefined();
    expect(typeof clientLoader).toBe("function");
  });
});
