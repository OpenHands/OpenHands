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

  const handleCreateCustomAutomation = () => {
    launchInChat(t(I18nKey.AUTOMATIONS$CUSTOM_AUTOMATION_PROMPT));
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
        className="flex flex-col gap-4 rounded-xl border border-[var(--oh-border)] bg-surface-raised p-4 lg:flex-row lg:items-center lg:justify-between"
      >
        <div className="min-w-0">
          <h2 className="text-base font-semibold text-content">
            {t(I18nKey.AUTOMATIONS$TEMPLATES_CTA_TITLE)}
          </h2>
          <p className="mt-1 text-sm leading-relaxed text-muted">
            {t(I18nKey.AUTOMATIONS$TEMPLATES_CTA_DESC)}
          </p>
        </div>
        <div className="flex shrink-0 flex-col gap-2 sm:flex-row sm:items-center">
          <BrandButton
            type="button"
            variant="primary"
            testId="automation-opportunities-cta-find"
            className="px-4 text-center"
            onClick={handleFindOpportunities}
          >
            {t(I18nKey.AUTOMATIONS$CREATE_AUTOMATION_BUTTON)}
          </BrandButton>
          <BrandButton
            type="button"
            variant="secondary"
            testId="automation-opportunities-cta-custom"
            className="px-4 text-center"
            onClick={handleCreateCustomAutomation}
          >
            {t(I18nKey.AUTOMATIONS$CUSTOM_AUTOMATION_BUTTON)}
          </BrandButton>
        </div>
      </section>
      <div className="flex max-w-xl items-stretch">
        <SearchInput value={searchQuery} onChange={setSearchQuery} />
      </div>
      <RecommendedAutomationsLauncher query={searchQuery} />
    </ManifestSubpageLayout>
  );
}
