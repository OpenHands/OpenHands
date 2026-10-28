import type { i18n as I18nInstance } from "i18next";
import i18n, { OPENHANDS_I18N_NAMESPACE } from "#/i18n";

/** Tree-scoped i18n, or the standalone proxy when a caller has no React instance. */
export type OpenHandsTranslator =
  | Pick<I18nInstance, "t" | "exists">
  | {
      t?: (key: string, options?: unknown) => unknown;
      exists?: (key: string, options?: unknown) => boolean;
    };

export function openhandsT(
  instance: OpenHandsTranslator | null | undefined,
  key: string,
): string {
  if (!instance || typeof instance.t !== "function") {
    return i18n.t(key);
  }

  return String(instance.t(key, { ns: OPENHANDS_I18N_NAMESPACE }));
}

export function openhandsExists(
  instance: OpenHandsTranslator | null | undefined,
  key: string,
): boolean {
  if (!instance || typeof instance.exists !== "function") {
    return i18n.exists(key);
  }

  return instance.exists(key, { ns: OPENHANDS_I18N_NAMESPACE });
}
