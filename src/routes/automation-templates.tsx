import { useState } from "react";
import { useTranslation } from "react-i18next";
import { useAutomationSubPageNav } from "#/components/features/automations/dashboard/use-automation-sub-page-nav";
import { RecommendedAutomationsLauncher } from "#/components/features/automations/recommended-automations-launcher";
import { SearchInput } from "#/components/features/automations/search-input";
import { BrandButton } from "#/components/features/settings/brand-button";
import { ManifestSubpageLayout } from "#/components/features/manifest/manifest-subpage-layout";
import { useLaunchSkillInChat } from "#/hooks/use-launch-skill-in-chat";
import { I18nKey } from "#/i18n/declaration";
import { getTemplatesPageSpec } from "#/manifests/automation-interface";

/**
 * The templates sub-page. It exists only while the admitted interface
 * manifest declares it — like a setup route for an id no entry claims, an
 * undeclared page is a 404 rendered by the layout's error boundary.
 */
export const clientLoader = () => {
  if (!getTemplatesPageSpec()) {
    throw new Response(null, { status: 404, statusText: "Not Found" });
  }
  return null;
};

export default function AutomationTemplates() {
  const { t } = useTranslation("openhands");
  const [searchQuery, setSearchQuery] = useState("");
  const spec = getTemplatesPageSpec();
  const nav = useAutomationSubPageNav();
  const launchInChat = useLaunchSkillInChat();

  if (!spec || !nav) return null;

  const handleFindOpportunities = () => {
    launchInChat(t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_PROMPT));
  };

  return (
    <ManifestSubpageLayout
      heading={nav.heading}
      navTestIdBase="automations-navbar"
      items={nav.items}
    >
      <div className="min-w-0">
        <h1 className="text-xl font-semibold text-content">{spec.title}</h1>
        <p className="mt-1 text-sm text-muted">{spec.description}</p>
      </div>
      <section
        data-testid="automation-opportunities-cta"
        className="flex flex-col gap-4 rounded-xl border border-[var(--oh-border)] bg-surface-raised p-4 sm:flex-row sm:items-center sm:justify-between"
      >
        <div className="min-w-0">
          <h2 className="text-base font-semibold text-content">
            {t(I18nKey.AUTOMATIONS$EMPTY_HOW_TO_CREATE_TITLE)}
          </h2>
          <p className="mt-1 text-sm leading-relaxed text-muted">
            {t(I18nKey.AUTOMATIONS$CREATE_INSTRUCTIONS_GUIDANCE)}
          </p>
        </div>
        <BrandButton
          type="button"
          variant="primary"
          testId="automation-opportunities-cta-button"
          onClick={handleFindOpportunities}
        >
          {t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
        </BrandButton>
      </section>
      <div className="flex max-w-xl items-stretch">
        <SearchInput value={searchQuery} onChange={setSearchQuery} />
      </div>
      <RecommendedAutomationsLauncher query={searchQuery} />
    </ManifestSubpageLayout>
  );
}
