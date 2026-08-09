import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Shield } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { SecurityFindingsPanel } from "#/components/features/security/security-findings-panel";
import { useConversationDependencyTrackIntegration } from "#/hooks/query/use-dependency-track-integration";
import { useSecuritySastScan } from "#/hooks/query/use-security-sast-scan";
import { useSecurityScaScan } from "#/hooks/query/use-security-sca-scan";
import type { ScaScanResult, SecurityScanResult } from "#/types/security-scan";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

export default function SecurityTab() {
  const { t } = useTranslation("openhands");
  const [lastSastResult, setLastSastResult] = useState<SecurityScanResult | null>(
    null,
  );
  const [lastScaResult, setLastScaResult] = useState<ScaScanResult | null>(null);
  const sastMutation = useSecuritySastScan();
  const scaMutation = useSecurityScaScan();
  const dtIntegration = useConversationDependencyTrackIntegration();

  const handleSastScan = async () => {
    try {
      const result = await sastMutation.mutateAsync();
      setLastSastResult(result);
    } catch {
      setLastSastResult(null);
    }
  };

  const handleScaScan = async () => {
    try {
      const result = await scaMutation.mutateAsync();
      setLastScaResult(result);
    } catch {
      setLastScaResult(null);
    }
  };

  const sastErrorMessage = (() => {
    if (!sastMutation.error) return null;
    switch (sastMutation.error.code) {
      case "opengrep_not_installed":
        return t(I18nKey.SECURITY$OPENGREP_NOT_INSTALLED);
      case "runtime_unavailable":
        return t(I18nKey.SECURITY$RUNTIME_UNAVAILABLE);
      case "invalid_output":
        return t(I18nKey.SECURITY$INVALID_OUTPUT);
      default:
        return t(I18nKey.SECURITY$SCAN_FAILED);
    }
  })();

  const scaErrorMessage = (() => {
    if (!scaMutation.error) return null;
    switch (scaMutation.error.code) {
      case "syft_not_installed":
        return t(I18nKey.SECURITY$SYFT_NOT_INSTALLED);
      case "dependency_track_not_configured":
        return t(I18nKey.SECURITY$DEPENDENCY_TRACK_NOT_CONFIGURED);
      case "bom_upload_failed":
        return t(I18nKey.SECURITY$SCA_UPLOAD_FAILED);
      case "bom_processing_failed":
        return t(I18nKey.SECURITY$SCA_PROCESSING_FAILED);
      case "runtime_unavailable":
        return t(I18nKey.SECURITY$RUNTIME_UNAVAILABLE);
      case "invalid_output":
        return t(I18nKey.SECURITY$INVALID_OUTPUT);
      default:
        return t(I18nKey.SECURITY$SCAN_FAILED);
    }
  })();

  return (
    <div className="flex h-full min-h-0 flex-col" data-testid="security-tab">
      <div className="shrink-0 border-b border-[var(--oh-border)] px-3 py-3">
        <div className="mb-3 flex items-center gap-2">
          <Shield className="h-4 w-4 text-[var(--oh-muted)]" aria-hidden />
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--oh-muted)]">
            {t(I18nKey.COMMON$SECURITY)}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button
            type="button"
            data-testid="security-sast-scan-button"
            onClick={() => void handleSastScan()}
            disabled={sastMutation.isPending}
            className={cn(
              "flex h-8 items-center justify-center rounded bg-white px-3 text-xs font-medium text-black transition-opacity",
              sastMutation.isPending
                ? "cursor-not-allowed opacity-60"
                : "cursor-pointer hover:opacity-90",
            )}
          >
            {sastMutation.isPending
              ? t(I18nKey.SECURITY$SCANNING)
              : t(I18nKey.SECURITY$SCAN_SAST)}
          </button>
          <button
            type="button"
            data-testid="security-sca-scan-button"
            onClick={() => void handleScaScan()}
            disabled={scaMutation.isPending || !dtIntegration.isReady}
            className={cn(
              "flex h-8 items-center justify-center rounded bg-white px-3 text-xs font-medium text-black transition-opacity",
              scaMutation.isPending || !dtIntegration.isReady
                ? "cursor-not-allowed opacity-60"
                : "cursor-pointer hover:opacity-90",
            )}
          >
            {scaMutation.isPending
              ? t(I18nKey.SECURITY$SCANNING)
              : t(I18nKey.SECURITY$SCAN_SCA)}
          </button>
          {(sastMutation.isPending || scaMutation.isPending) && (
            <LoadingSpinner size="small" data-testid="security-scan-spinner" />
          )}
        </div>
        {!dtIntegration.isReady && !dtIntegration.isLoading && (
          <p
            className="mt-2 text-xs text-[var(--oh-muted)]"
            data-testid="security-sca-config-hint"
          >
            {t(I18nKey.SECURITY$DEPENDENCY_TRACK_NOT_CONFIGURED)}
          </p>
        )}
        {sastErrorMessage && (
          <p
            className="mt-2 text-xs text-red-400"
            data-testid="security-sast-scan-error"
            role="alert"
          >
            {sastErrorMessage}
          </p>
        )}
        {scaErrorMessage && (
          <p
            className="mt-2 text-xs text-red-400"
            data-testid="security-sca-scan-error"
            role="alert"
          >
            {scaErrorMessage}
          </p>
        )}
        {!sastMutation.isPending && lastSastResult && !sastMutation.isError && (
          <p className="mt-2 text-xs text-[var(--oh-muted)]">
            {t(I18nKey.SECURITY$SAST_SCAN_COMPLETE)}
          </p>
        )}
        {!scaMutation.isPending && lastScaResult && !scaMutation.isError && (
          <p className="mt-2 text-xs text-[var(--oh-muted)]">
            {t(I18nKey.SECURITY$SCA_SCAN_COMPLETE)}
          </p>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        <SecurityFindingsPanel
          sastResult={lastSastResult}
          scaResult={lastScaResult}
        />
      </div>
    </div>
  );
}
