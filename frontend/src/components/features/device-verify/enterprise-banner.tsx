import { useTranslation } from "react-i18next";
import { usePostHog } from "posthog-js/react";
import { I18nKey } from "#/i18n/declaration";

const ENTERPRISE_FEATURE_KEYS: I18nKey[] = [
  I18nKey.ENTERPRISE$FEATURE_DATA_PRIVACY,
  I18nKey.ENTERPRISE$FEATURE_DEPLOYMENT,
  I18nKey.ENTERPRISE$FEATURE_SSO,
  I18nKey.ENTERPRISE$FEATURE_SUPPORT,
];

export function EnterpriseBanner() {
  const { t } = useTranslation();
  const posthog = usePostHog();

  const handleLearnMore = () => {
    posthog.capture("saas_selfhosted_inquiry");
    window.open("https://openhands.dev/enterprise", "_blank", "noopener");
  };

  return (
    <div className="w-full max-w-md mx-auto lg:mx-0 lg:w-80 p-6 rounded-lg bg-gradient-to-b from-slate-800 to-slate-900 border border-slate-700">
      {/* Self-Hosted Label */}
      <div className="flex justify-center mb-4">
        <div className="px-4 py-1 rounded-full bg-gradient-to-r from-blue-900 to-blue-950 border border-blue-800">
          <span className="text-xs font-medium text-blue-400 tracking-wider uppercase">
            {t(I18nKey.ENTERPRISE$SELF_HOSTED)}
          </span>
        </div>
      </div>

      {/* Title */}
      <h2 className="text-xl font-semibold text-white text-center mb-3">
        {t(I18nKey.ENTERPRISE$TITLE)}
      </h2>

      {/* Description */}
      <p className="text-sm text-gray-400 text-center mb-6">
        {t(I18nKey.ENTERPRISE$DESCRIPTION)}
      </p>

      {/* Features List */}
      <ul className="space-y-3 mb-6">
        {ENTERPRISE_FEATURE_KEYS.map((featureKey) => (
          <li key={featureKey} className="flex items-center gap-2">
            <svg
              className="w-4 h-4 text-blue-400 flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-sm text-gray-300">{t(featureKey)}</span>
          </li>
        ))}
      </ul>

      {/* Learn More Button */}
      <button
        type="button"
        onClick={handleLearnMore}
        className="w-full py-2.5 px-4 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium transition-colors"
      >
        {t(I18nKey.ENTERPRISE$LEARN_MORE)}
      </button>
    </div>
  );
}
