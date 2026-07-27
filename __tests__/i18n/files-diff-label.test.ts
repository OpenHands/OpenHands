import { describe, expect, it } from "vitest";
import fs from "fs";
import path from "path";

// The Files-tab diff toggle was renamed from "Diff view" to just "Diff".
// Lock that down at the source-of-truth (translation.json) rather than the
// rendered label, because the test environment's i18next mock returns keys
// rather than translated strings.
describe("FILES$DIFF_VIEW label", () => {
  const translationPath = path.join(
    __dirname,
    "../../src/i18n/translation.json",
  );
  const translation = JSON.parse(
    fs.readFileSync(translationPath, "utf-8"),
  ) as Record<string, Record<string, string>>;

  it('uses "Diff" (not "Diff view") in English', () => {
    expect(translation.FILES$DIFF_VIEW).toBeDefined();
    expect(translation.FILES$DIFF_VIEW.en).toBe("Diff");
  });
});
