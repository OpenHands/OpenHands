import fs from "fs";
import path from "path";
import { describe, expect, it } from "vitest";
import { AvailableLanguages } from "#/i18n";

// The sidebar's "Search commands" trigger is the command menu's accessible
// name, so it must be read out in the user's language (#17909). Lock this down
// at translation.json because the test i18next mock returns keys.
describe("COMMAND_MENU$OPEN_LABEL label", () => {
  const translationPath = path.join(
    __dirname,
    "../../src/i18n/translation.json",
  );
  const translation = JSON.parse(
    fs.readFileSync(translationPath, "utf-8"),
  ) as Record<string, Record<string, string>>;
  const label = translation.COMMAND_MENU$OPEN_LABEL;

  it.each(
    AvailableLanguages.map(({ value }) => value).filter((lng) => lng !== "en"),
  )("is translated for %s", (lng) => {
    expect(label[lng]).toBeTruthy();
    expect(label[lng]).not.toBe(label.en);
  });
});
