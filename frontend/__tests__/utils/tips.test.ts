import { describe, expect, it, vi, afterEach } from "vitest";
import { I18nKey } from "#/i18n/declaration";
import { ProviderOptions } from "#/types/settings";
import { getAvailableTipsForProvider, getRandomTip } from "#/utils/tips";

describe("tips", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("filters out GitHub-only tips when the current conversation uses GitLab", () => {
    const tips = getAvailableTipsForProvider(ProviderOptions.gitlab);

    expect(tips.find((tip) => tip.key === I18nKey.TIPS$GITHUB_HOOK)).toBe(
      undefined,
    );
  });

  it("keeps GitHub-only tips when the current conversation uses GitHub", () => {
    const tips = getAvailableTipsForProvider(ProviderOptions.github);

    expect(tips.find((tip) => tip.key === I18nKey.TIPS$GITHUB_HOOK)).toBeTruthy();
  });

  it("never returns a GitHub-only tip for GitLab conversations", () => {
    vi.spyOn(Math, "random").mockReturnValue(0.999999);

    const tip = getRandomTip(ProviderOptions.gitlab);

    expect(tip.key).not.toBe(I18nKey.TIPS$GITHUB_HOOK);
  });
});
