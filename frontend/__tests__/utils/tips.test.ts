import { describe, expect, it, vi, afterEach } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { ProviderOptions } from "#/types/settings";
import { getAvailableTips, getRandomTip } from "#/utils/tips";

describe("tips", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("filters out GitHub-only tips when the user does not have GitHub configured", () => {
    const tips = getAvailableTips([ProviderOptions.gitlab]);

    expect(tips.find((tip) => tip.key === I18nKey.TIPS$GITHUB_HOOK)).toBe(
      undefined,
    );
  });

  it("keeps GitHub-only tips when the user has GitHub configured", () => {
    const tips = getAvailableTips([ProviderOptions.github]);

    expect(tips.find((tip) => tip.key === I18nKey.TIPS$GITHUB_HOOK)).toBeTruthy();
  });

  it("never returns a GitHub-only tip for users without GitHub access", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.999999);

    const tip = getRandomTip([ProviderOptions.gitlab]);

    expect(tip.key).not.toBe(I18nKey.TIPS$GITHUB_HOOK);
  });
});
