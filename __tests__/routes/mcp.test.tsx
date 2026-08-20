import { describe, expect, it } from "vitest";
import * as mcpRoute from "#/routes/mcp";

describe("mcp route", () => {
  it("does not gate the MCP page behind an ACP redirect", () => {
    expect("clientLoader" in mcpRoute).toBe(false);
  });
});
