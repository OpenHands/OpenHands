import { getActiveBackend } from "#/api/backend-registry/active-store";
import type { ResolvedActiveBackend } from "#/api/backend-registry/types";
import i18n from "#/i18n";
import { I18nKey } from "#/i18n/declaration";

export function assertActiveEnvironment(expected: ResolvedActiveBackend): void {
  const current = getActiveBackend();
  if (
    expected.backend.id !== current.backend.id ||
    expected.backend.connectionRevision !==
      current.backend.connectionRevision ||
    expected.orgId !== current.orgId
  ) {
    throw new Error(i18n.t(I18nKey.CONVERSATION$START_ENVIRONMENT_CHANGED));
  }
}
