import type { I18nKey } from "#/i18n/declaration";

// @spec AWI-001 — Matched routes own their section title
export interface RouteTitleHandle {
  titleKey?: I18nKey;
  titleSource?: "automation" | "extension";
}

export const AUTOMATION_TITLE_HANDLE: RouteTitleHandle = {
  titleSource: "automation",
};
