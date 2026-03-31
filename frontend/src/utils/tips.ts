import { I18nKey } from "#/i18n/declaration";
import { ProviderOptions, type Provider } from "#/types/settings";

export interface Tip {
  key: I18nKey;
  link?: string;
  providers?: Provider[];
}

export const TIPS: Tip[] = [
  {
    key: I18nKey.TIPS$CUSTOMIZE_MICROAGENT,
    link: "https://docs.all-hands.dev/usage/prompting/microagents-repo",
  },
  {
    key: I18nKey.TIPS$SETUP_SCRIPT,
    link: "https://docs.all-hands.dev/usage/prompting/repository#setup-script",
  },
  { key: I18nKey.TIPS$VSCODE_INSTANCE },
  { key: I18nKey.TIPS$SAVE_WORK },
  {
    key: I18nKey.TIPS$SPECIFY_FILES,
    link: "https://docs.all-hands.dev/usage/prompting/prompting-best-practices",
  },
  {
    key: I18nKey.TIPS$HEADLESS_MODE,
    link: "https://docs.all-hands.dev/usage/how-to/headless-mode",
  },
  {
    key: I18nKey.TIPS$CLI_MODE,
    link: "https://docs.all-hands.dev/usage/how-to/cli-mode",
  },
  {
    key: I18nKey.TIPS$GITHUB_HOOK,
    link: "https://docs.all-hands.dev/usage/cloud/github-installation#working-on-github-issues-and-pull-requests-using-openhands",
    providers: [ProviderOptions.github],
  },
  {
    key: I18nKey.TIPS$BLOG_SIGNUP,
    link: "https://www.all-hands.dev/blog",
  },
  {
    key: I18nKey.TIPS$API_USAGE,
    link: "https://docs.all-hands.dev/api-reference/health-check",
  },
];

export function getAvailableTipsForProvider(
  currentProvider: Provider | null = null,
): Tip[] {
  return TIPS.filter(
    (tip) =>
      !tip.providers ||
      (currentProvider !== null && tip.providers.includes(currentProvider)),
  );
}

export function getRandomTip(currentProvider: Provider | null = null): Tip {
  const availableTips = getAvailableTipsForProvider(currentProvider);
  const tips = availableTips.length > 0 ? availableTips : TIPS;
  const randomIndex = Math.floor(Math.random() * tips.length);
  return tips[randomIndex];
}
