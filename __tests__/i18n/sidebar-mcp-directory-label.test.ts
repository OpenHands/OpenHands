import fs from "fs";
import path from "path";
import { describe, expect, it } from "vitest";

// The sidebar MCP entry must not use the ambiguous "Integrations" label.
// Lock this down at translation.json because the test i18next mock returns keys.
describe("SIDEBAR$MCP_DIRECTORY label", () => {
  const translationPath = path.join(
    __dirname,
    "../../src/i18n/translation.json",
  );
  const translation = JSON.parse(
    fs.readFileSync(translationPath, "utf-8"),
  ) as Record<string, Record<string, string>>;

  it('uses "MCP Directory" (not "Integrations") in English', () => {
    expect(translation.SIDEBAR$MCP_DIRECTORY).toBeDefined();
    expect(translation.SIDEBAR$MCP_DIRECTORY.en).toBe("MCP Directory");
    expect(translation.SIDEBAR$MCP_DIRECTORY.en).not.toBe("Integrations");
  });
});
