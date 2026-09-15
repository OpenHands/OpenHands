import { useTranslation } from "react-i18next";
import { FaTriangleExclamation } from "react-icons/fa6";
import { I18nKey } from "#/i18n/declaration";
import { useNavigation } from "#/context/navigation-context";
import { useLlmConfigured } from "#/hooks/use-llm-configured";
import { BrandButton } from "#/components/features/settings/brand-button";
import { Typography } from "#/ui/typography";

/**
 * Warns the user on the home screen when the active agent has no usable LLM —
 * most notably after they skip onboarding, which persists no settings. Offers
 * a single action that routes to LLM settings so the failure is communicated
 * up front instead of surfacing only when a conversation attempt errors out.
 *
 * Renders nothing while settings load (avoids a flash) or once the LLM is
 * configured; the settings query refetches after a key is saved, so the banner
 * unmounts on its own.
 */
export function LlmNotConfiguredBanner() {
  const { t } = useTranslation("openhands");
  const { navigate } = useNavigation();
  const { isConfigured, isLoading, apiKeyMissing } = useLlmConfigured();

  if (isLoading || isConfigured) {
    return null;
  }

  // #15609: when an active local profile exists but its API key is missing,
  // the generic "LLM not set up" message is misleading — the user already
  // set up a profile, the key just got lost. Surface the specific cause and
  // route them straight to the LLM settings where they can re-enter it.
  const messageKey = apiKeyMissing
    ? I18nKey.HOME$LLM_API_KEY_MISSING_MESSAGE
    : I18nKey.HOME$LLM_NOT_CONFIGURED_MESSAGE;
  const actionKey = apiKeyMissing
    ? I18nKey.HOME$LLM_API_KEY_MISSING_ACTION
    : I18nKey.HOME$LLM_NOT_CONFIGURED_ACTION;

  return (
    <div
      data-testid="home-llm-not-configured-banner"
      role="alert"
      className="mt-3 flex w-full flex-col gap-3 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] px-4 py-3 text-[var(--oh-foreground)] sm:flex-row sm:items-center sm:justify-between sm:py-4"
    >
      <div className="flex min-w-0 items-start sm:items-center">
        <div className="flex-shrink-0">
          <FaTriangleExclamation className="align-middle text-yellow-400" />
        </div>
        <Typography.Text className="ml-3 text-sm font-medium">
          {t(messageKey)}
        </Typography.Text>
      </div>

      <BrandButton
        testId="home-llm-not-configured-action"
        type="button"
        variant="primary"
        className="w-fit shrink-0 self-start whitespace-nowrap sm:self-auto"
        onClick={() => navigate("/settings/llm")}
      >
        {t(actionKey)}
      </BrandButton>
    </div>
  );
}
