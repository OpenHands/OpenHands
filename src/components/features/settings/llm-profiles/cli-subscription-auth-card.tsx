import { useQueryClient } from "@tanstack/react-query";
import { useTranslation } from "react-i18next";
import { BrandButton } from "#/components/features/settings/brand-button";
import { AcpAuthStatusBanner } from "#/components/features/settings/acp-auth-status-banner";
import { useAcpAuthStatus } from "#/hooks/query/use-acp-auth-status";
import { I18nKey } from "#/i18n/declaration";
import { Typography } from "#/ui/typography";

interface CliSubscriptionAuthCardProps {
  probeKey: string;
  providerName: string;
  loginCommand: string;
}

/**
 * Host-CLI subscription login for Add provider (Claude / Cursor / OpenCode).
 * The agent-server has no device-flow for these — we probe the CLI the same
 * way Settings → Agent probes Claude Code.
 */
export function CliSubscriptionAuthCard({
  probeKey,
  providerName,
  loginCommand,
}: CliSubscriptionAuthCardProps) {
  const { t } = useTranslation("openhands");
  const queryClient = useQueryClient();
  const { status, isChecking } = useAcpAuthStatus(probeKey);

  return (
    <div
      data-testid="cli-subscription-auth-card"
      className="flex flex-col gap-3"
    >
      <AcpAuthStatusBanner
        status={status}
        isChecking={isChecking}
        providerName={providerName}
        testIdPrefix="cli-subscription-auth"
      />
      <Typography.Text className="text-sm text-[var(--oh-muted)]">
        {t(I18nKey.SETTINGS$CLI_SUBSCRIPTION_LOGIN_HINT, {
          command: loginCommand,
        })}
      </Typography.Text>
      <BrandButton
        type="button"
        variant="tertiary"
        testId="cli-subscription-recheck"
        onClick={() => {
          queryClient.invalidateQueries({
            queryKey: ["acp-auth-status"],
          });
        }}
      >
        {t(I18nKey.SETTINGS$CLI_SUBSCRIPTION_RECHECK)}
      </BrandButton>
    </div>
  );
}
