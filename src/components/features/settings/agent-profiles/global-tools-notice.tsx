import { useTranslation } from "react-i18next";
import { AxiosError } from "axios";
import { BrandButton } from "#/components/features/settings/brand-button";
import { useSettings } from "#/hooks/query/use-settings";
import { useSaveSettings } from "#/hooks/mutation/use-save-settings";
import { readProfileTools } from "#/constants/profile-tools";
import { Typography } from "#/ui/typography";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import { settingsListContainerClassName } from "#/utils/settings-list-classes";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";
import { retrieveAxiosErrorMessage } from "#/utils/retrieve-axios-error-message";

/** Surface a custom global `tools` list, which no editor shows, and reset it. */
export function GlobalToolsNotice() {
  const { t } = useTranslation("openhands");
  const { data: settings } = useSettings();
  const { mutate: saveSettings, isPending } = useSaveSettings();

  const agentSettings = settings?.agent_settings;
  if (!agentSettings || agentSettings.agent_kind === "acp") return null;
  const tools = readProfileTools(agentSettings.tools);
  if (tools.mode !== "custom") return null;

  const handleReset = () =>
    saveSettings(
      { agent_settings_diff: { tools: null } },
      {
        onError: (error) => {
          const message = retrieveAxiosErrorMessage(error as AxiosError);
          displayErrorToast(message || t(I18nKey.ERROR$GENERIC));
        },
        onSuccess: () => displaySuccessToast(t(I18nKey.SETTINGS$SAVED)),
      },
    );

  return (
    <div
      data-testid="global-tools-notice"
      className={cn(
        settingsListContainerClassName,
        "flex flex-wrap items-center justify-between gap-3 p-3",
      )}
    >
      <Typography.Text className="text-sm">
        {t(I18nKey.SETTINGS$GLOBAL_TOOLS_CUSTOM, {
          names: tools.selected.join(", ") || "—",
        })}
      </Typography.Text>
      <BrandButton
        testId="global-tools-reset"
        type="button"
        variant="secondary"
        isDisabled={isPending}
        onClick={handleReset}
      >
        {t(I18nKey.SETTINGS$GLOBAL_TOOLS_RESET)}
      </BrandButton>
    </div>
  );
}
