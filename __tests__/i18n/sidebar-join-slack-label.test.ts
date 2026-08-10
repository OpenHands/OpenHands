import fs from "fs";
import path from "path";
import { describe, expect, it } from "vitest";

// The sidebar Join Slack entry renders `t(I18nKey.SIDEBAR$JOIN_SLACK)`, and
// `I18nKey` is generated from translation.json by make-i18n-translations.cjs.
// A key missing here therefore compiles to `undefined` and renders an icon
// with no label. Component tests cannot catch that, because the test i18next
// mock returns keys rather than resolved translations, so lock it down at the
// source file the generator actually reads.
describe("SIDEBAR$JOIN_SLACK label", () => {
  const translationPath = path.join(
    __dirname,
    "../../src/i18n/translation.json",
  );
  const translation = JSON.parse(
    fs.readFileSync(translationPath, "utf-8"),
  ) as Record<string, Record<string, string>>;

  it('is registered in the generator source as "Join Slack"', () => {
    expect(translation.SIDEBAR$JOIN_SLACK).toBeDefined();
    expect(translation.SIDEBAR$JOIN_SLACK.en).toBe("Join Slack");
  });

  it("covers every locale the other sidebar entries cover", () => {
    const locales = Object.keys(translation.SIDEBAR$NAVIGATION_LABEL);
    for (const locale of locales) {
      expect(translation.SIDEBAR$JOIN_SLACK[locale]?.trim()).toBeTruthy();
    }
  });
});
