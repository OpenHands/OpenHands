import { useTranslation } from "react-i18next";
import { SettingsSwitch } from "#/components/features/settings/settings-switch";
import { SwitchSkeleton } from "#/components/features/settings/switch-skeleton";
import { useFeatures } from "#/hooks/query/use-features";
import { useUpdateFeature } from "#/hooks/mutation/use-update-feature";
import { I18nKey } from "#/i18n/declaration";
import { displayApiErrorToast } from "#/utils/custom-toast-handlers";

/**
 * Features the user can turn on or off for themselves. The list, labels and
 * descriptions come from the agent server (`GET /api/settings/features`), so a
 * new feature needs no Canvas change to appear here.
 */
function FeaturesSettingsScreen() {
  const { t } = useTranslation("openhands");
  const { data: features, isLoading, isError } = useFeatures();
  const { mutate: updateFeature, isPending } = useUpdateFeature();

  if (isLoading) {
    return (
      <div data-testid="features-settings-screen">
        <SwitchSkeleton />
      </div>
    );
  }

  if (isError) {
    return (
      <div data-testid="features-settings-screen">
        <p className="text-sm text-danger">
          {t(I18nKey.SETTINGS$FEATURES_LOAD_FAILED)}
        </p>
      </div>
    );
  }

  if (!features?.length) {
    return (
      <div data-testid="features-settings-screen">
        <p className="text-sm text-muted">
          {t(I18nKey.SETTINGS$FEATURES_EMPTY)}
        </p>
      </div>
    );
  }

  return (
    <div data-testid="features-settings-screen" className="flex flex-col gap-6">
      {features.map((feature) => (
        <div key={feature.key} className="flex flex-col gap-1">
          <SettingsSwitch
            testId={`feature-switch-${feature.key}`}
            name={`feature-switch-${feature.key}`}
            isToggled={feature.enabled}
            isDisabled={isPending}
            onToggle={(enabled) =>
              updateFeature(
                { key: feature.key, enabled },
                {
                  onError: (error) =>
                    displayApiErrorToast(
                      error,
                      t(I18nKey.SETTINGS$FEATURES_UPDATE_FAILED),
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
    </div>
  );
}

export default FeaturesSettingsScreen;
