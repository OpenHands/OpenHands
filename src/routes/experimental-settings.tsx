import { useTranslation } from "react-i18next";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SwitchSkeleton } from "#/components/features/settings/switch-skeleton";
import { useExperimentalFeatures } from "#/hooks/query/use-experimental-features";
import { useUpdateExperimentalFeature } from "#/hooks/mutation/use-update-experimental-feature";
import { I18nKey } from "#/i18n/declaration";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Experimental features the user can turn on or off for themselves. The list, labels and
 * descriptions come from the agent server (`GET /api/settings/experimental`), so a
 * new experimental feature needs no Canvas change to appear here.
 */
function ExperimentalSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { data: features, isLoading, isError } = useExperimentalFeatures();
  const { mutate: updateExperimentalFeature, isPending } =
    useUpdateExperimentalFeature();

  if (isLoading) {
    return (
      <div data-testid="experimental-settings-screen">
        <SwitchSkeleton />
      </div>
    );
  }

  if (isError) {
    return (
      <div data-testid="experimental-settings-screen">
        <p className="text-sm text-danger">
          {t(I18nKey.SETTINGS$EXPERIMENTAL_LOAD_FAILED)}
        </p>
      </div>
    );
  }

  if (!features?.length) {
    return (
      <div data-testid="experimental-settings-screen">
        <p className="text-sm text-muted">
          {t(I18nKey.SETTINGS$EXPERIMENTAL_EMPTY)}
        </p>
      </div>
    );
  }

  return (
    <div
      data-testid="experimental-settings-screen"
      className="flex flex-col gap-6"
    >
      {features.map((feature) => (
        <div key={feature.key} className="flex flex-col gap-1">
          <SettingsSwitch
            testId={`experimental-switch-${feature.key}`}
            name={`experimental-switch-${feature.key}`}
            isToggled={feature.enabled}
            isDisabled={isPending}
            onToggle={(enabled) =>
              updateExperimentalFeature(
                { key: feature.key, enabled },
                {
                  onError: (error) =>
                    displayApiErrorToast(
                      error,
                      t(I18nKey.SETTINGS$EXPERIMENTAL_UPDATE_FAILED),
                    ),
                },
              )
            }
          >
            {feature.label}
          </SettingsSwitch>
          <p className="max-w-prose pl-12 text-xs text-muted">
            {feature.description}
          </p>
        </div>
      ))}
      <p
        data-testid="experimental-notice"
        className="max-w-prose border-t border-border pt-4 text-xs text-muted"
      >
        {t(I18nKey.SETTINGS$EXPERIMENTAL_NOTICE)}
      </p>
    </div>
  );
}

export default ExperimentalSettingsScreen;
