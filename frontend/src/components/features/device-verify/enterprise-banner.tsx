/* eslint-disable i18next/no-literal-string */
import { usePostHog } from "posthog-js/react";

const ENTERPRISE_FEATURES = [
  "Full data privacy & control",
  "Custom deployment options",
  "SSO & enterprise auth",
  "Dedicated support",
];

export function EnterpriseBanner() {
  const posthog = usePostHog();

  const handleLearnMore = () => {
    posthog.capture("saas_selfhosted_inquiry");
    window.open("https://openhands.dev/enterprise", "_blank", "noopener");
  };

  return (
    <div className="w-full max-w-md mx-auto lg:mx-0 lg:w-80 p-6 rounded-lg bg-gradient-to-b from-[#1a2744] to-[#0d1829] border border-[#2a3f5f]">
      {/* Self-Hosted Label */}
      <div className="flex justify-center mb-4">
        <div className="px-4 py-1 rounded-full bg-gradient-to-r from-[#1a4a6e] to-[#1a3a5e] border border-[#2a5a8e]">
          <span className="text-xs font-medium text-[#4a9eff] tracking-wider uppercase">
            Self-Hosted
          </span>
        </div>
      </div>

      {/* Title */}
      <h2 className="text-xl font-semibold text-white text-center mb-3">
        OpenHands Enterprise
      </h2>

      {/* Description */}
      <p className="text-sm text-gray-400 text-center mb-6">
        Complete data control with your own self-hosted AI development platform.
      </p>

      {/* Features List */}
      <ul className="space-y-3 mb-6">
        {ENTERPRISE_FEATURES.map((feature) => (
          <li key={feature} className="flex items-center gap-2">
            <svg
              className="w-4 h-4 text-[#4a9eff] flex-shrink-0"
              fill="currentColor"
              viewBox="0 0 20 20"
            >
              <path
                fillRule="evenodd"
                d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z"
                clipRule="evenodd"
              />
            </svg>
            <span className="text-sm text-gray-300">{feature}</span>
          </li>
        ))}
      </ul>

      {/* Learn More Button */}
      <button
        type="button"
        onClick={handleLearnMore}
        className="w-full py-2.5 px-4 rounded-lg bg-[#2563eb] hover:bg-[#1d4ed8] text-white font-medium transition-colors"
      >
        Learn More
      </button>
    </div>
  );
}
