import { useTranslation } from "react-i18next";
import { usePostHog } from "posthog-js/react";
import { I18nKey } from "#/i18n/declaration";
import { H2, Text } from "#/ui/typography";
import CheckCircleFillIcon from "#/icons/check-circle-fill.svg?react";
import { ENABLE_PROJ_USER_JOURNEY } from "#/utils/feature-flags";

const ENTERPRISE_FEATURE_KEYS: I18nKey[] = [
  I18nKey.ENTERPRISE$FEATURE_DATA_PRIVACY,
  I18nKey.ENTERPRISE$FEATURE_DEPLOYMENT,
  I18nKey.ENTERPRISE$FEATURE_SSO,
  I18nKey.ENTERPRISE$FEATURE_SUPPORT,
];

export function EnterpriseBanner() {
  const { t } = useTranslation();
  const posthog = usePostHog();

  if (!ENABLE_PROJ_USER_JOURNEY()) {
    return null;
  }

  const handleLearnMore = () => {
    posthog?.capture("saas_selfhosted_inquiry");
  };

  return (
    <div className="w-full max-w-md mx-auto lg:mx-0 lg:w-80 p-6 rounded-lg bg-gradient-to-b from-slate-800 to-slate-900 border border-slate-700 h-fit">
      {/* Self-Hosted Label */}
      <div className="flex justify-center mb-4">
        <div className="px-8 py-0.5 rounded-full bg-gradient-to-r from-blue-900 to-blue-950 border border-blue-800">
          <Text className="text-xs font-medium text-blue-400 tracking-wider uppercase">
            {t(I18nKey.ENTERPRISE$SELF_HOSTED)}
          </Text>
        </div>
      </div>

      {/* Title */}
      <H2 className="text-center mb-3">{t(I18nKey.ENTERPRISE$TITLE)}</H2>

      {/* Description */}
      <Text className="text-sm text-gray-400 text-center mb-6 block">
        {t(I18nKey.ENTERPRISE$DESCRIPTION)}
      </Text>

      {/* Features List */}
      <ul className="space-y-3 mb-6">
        {ENTERPRISE_FEATURE_KEYS.map((featureKey) => (
          <li key={featureKey} className="flex items-center gap-2">
            <CheckCircleFillIcon className="w-4 h-4 text-blue-400 flex-shrink-0" />
            <Text className="text-sm text-gray-300">{t(featureKey)}</Text>
          </li>
        ))}
      </ul>

      {/* Learn More Button */}
      <a
        href="https://openhands.dev/enterprise"
        target="_blank"
        rel="noopener noreferrer"
        onClick={handleLearnMore}
        aria-label={t(I18nKey.ENTERPRISE$LEARN_MORE_ARIA)}
        className="block w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors text-center"
      >
        {t(I18nKey.ENTERPRISE$LEARN_MORE)}
      </a>
    </div>
  );
}
