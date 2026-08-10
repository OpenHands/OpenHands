/**
 * Connect-a-Provider wizard (scaffold, draft).
 *
 * Flow: Vendor + key -> auto test-connection -> pick models -> confirm.
 * Writes a connection + bulk-creates N LLM profiles that reference the
 * connection's secret BY NAME (no inline key duplication).
 *
 * Reuses existing: useSearchProviders, useProviderModels, useVerifiedModels,
 * ModelSelector (verified/unverified split), llm-subscription-service precedent.
 *
 * Tracking: OpenHands/OpenHands#15492, Linear OSS-5295.
 * Scope: OpenHands frontend PR3 of the provider-connections plan.
 *
 * TODO (implementation):
 *  - Wizard steps component under src/components/shared/modals/settings/.
 *  - Bulk actions: Select all verified / Select all / Clear.
 *  - "More from {vendor}" collapsible for experimental models.
 *  - Confirmation summary ("Adding N LLM profiles from {vendor}. One key.").
 *  - Consumes @openhands/typescript-client connectionService (after release gate).
 */

export function ConnectProviderWizard() {
  // TODO
  return null;
}
