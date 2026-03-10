import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { useConfig } from "#/hooks/query/use-config";
import { useAuthUrl } from "#/hooks/use-auth-url";
import { BrandButton } from "../brand-button";

export function ConfigureAzureDevOpsAnchor() {
  const { t } = useTranslation();
  const { data: config } = useConfig();

  const authUrl = useAuthUrl({
    appMode: config?.app_mode ?? null,
    identityProvider: "azure_devops",
    authUrl: config?.auth_url,
  });

  const handleOAuthFlow = () => {
    if (!authUrl) {
      return;
    }

    window.location.href = authUrl;
  };

  return (
    <div data-testid="configure-azure-devops-button" className="py-9">
      <BrandButton
        type="button"
        variant="primary"
        className="w-55"
        onClick={handleOAuthFlow}
      >
        {t(I18nKey.AZURE_DEVOPS$CONNECT_ACCOUNT)}
      </BrandButton>
    </div>
  );
}
