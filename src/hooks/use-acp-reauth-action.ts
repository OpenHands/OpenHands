import { useNavigate } from "react-router";
import { useTranslation } from "react-i18next";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { I18nKey } from "#/i18n/declaration";
import { isAcpAuthErrorCode } from "#/utils/acp-error-codes";
import { cloudSettingsUrl } from "#/utils/cloud-integrations-url";

// The cloud app keeps this URL working: it redirects to the Agent Profiles
// page when that page replaces the single-agent page.
const CLOUD_AGENT_SETTINGS_PATH = "/settings/agent";

export interface AcpReauthAction {
  onReauth?: () => void;
  link?: { href: string; label: string; openInNewTab: boolean };
}

/**
 * Where the error banner sends the user to fix an ACP credential failure.
 * Locked-to-Cloud blocks the Canvas agent settings page, so the cloud's own
 * agent settings page, on the same host, is used instead.
 */
export function useAcpReauthAction(errorCode: string | null): AcpReauthAction {
  const { t } = useTranslation("openhands");
  const navigate = useNavigate();
  const { backend, orgId } = useActiveBackend();

  if (!isAcpAuthErrorCode(errorCode)) return {};
  if (getLockedCloudHost() === null) {
    return { onReauth: () => navigate("/settings/agents") };
  }
  if (backend.kind !== "cloud") return {};
  return {
    link: {
      href: cloudSettingsUrl(backend, CLOUD_AGENT_SETTINGS_PATH, orgId),
      label: t(I18nKey.ERROR$ACP_UPDATE_CREDENTIALS),
      openInNewTab: false,
    },
  };
}
