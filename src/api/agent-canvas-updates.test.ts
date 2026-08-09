import { describe, expect, it } from "vitest";
import { fetchLatestAgentCanvasVersion } from "./agent-canvas-updates";

describe("fetchLatestAgentCanvasVersion", () => {
  it("rejects without calling the npm registry (Heimdall fork)", async () => {
    await expect(fetchLatestAgentCanvasVersion()).rejects.toThrow(
      "disabled in this fork",
    );
  });
});
