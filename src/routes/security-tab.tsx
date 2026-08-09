import { useState } from "react";
import { useTranslation } from "react-i18next";
import { Shield } from "lucide-react";
import { I18nKey } from "#/i18n/declaration";
import { SecurityScanResults } from "#/components/features/security/security-scan-results";
import { useSecuritySastScan } from "#/hooks/query/use-security-sast-scan";
import type { SecurityScanResult } from "#/types/security-scan";
import { cn } from "#/utils/utils";
import { LoadingSpinner } from "#/components/shared/loading-spinner";

export default function SecurityTab() {
  const { t } = useTranslation("openhands");
  const [lastResult, setLastResult] = useState<SecurityScanResult | null>(null);
  const scanMutation = useSecuritySastScan();

  const handleScan = async () => {
    try {
      const result = await scanMutation.mutateAsync();
      setLastResult(result);
    } catch {
      setLastResult(null);
    }
  };

  const errorMessage = (() => {
    if (!scanMutation.error) return null;
    switch (scanMutation.error.code) {
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

  return (
    <div
      className="flex h-full min-h-0 flex-col"
      data-testid="security-tab"
    >
      <div className="shrink-0 border-b border-[var(--oh-border)] px-3 py-3">
        <div className="mb-3 flex items-center gap-2">
          <Shield className="h-4 w-4 text-[var(--oh-muted)]" aria-hidden />
          <p className="text-xs font-semibold uppercase tracking-wide text-[var(--oh-muted)]">
            {t(I18nKey.COMMON$SECURITY)}
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            data-testid="security-scan-button"
            onClick={() => void handleScan()}
            disabled={scanMutation.isPending}
            className={cn(
              "flex h-8 items-center justify-center rounded bg-white px-3 text-xs font-medium text-black transition-opacity",
              scanMutation.isPending
                ? "cursor-not-allowed opacity-60"
                : "cursor-pointer hover:opacity-90",
            )}
          >
            {scanMutation.isPending
              ? t(I18nKey.SECURITY$SCANNING)
              : t(I18nKey.SECURITY$SCAN)}
          </button>
          {scanMutation.isPending && (
            <LoadingSpinner size="small" data-testid="security-scan-spinner" />
          )}
        </div>
        {errorMessage && (
          <p
            className="mt-2 text-xs text-red-400"
            data-testid="security-scan-error"
            role="alert"
          >
            {errorMessage}
          </p>
        )}
        {!scanMutation.isPending && lastResult && !scanMutation.isError && (
          <p className="mt-2 text-xs text-[var(--oh-muted)]">
            {t(I18nKey.SECURITY$SCAN_COMPLETE)}
          </p>
        )}
      </div>
      <div className="min-h-0 flex-1 overflow-auto p-3">
        <p className="mb-3 text-xs font-medium text-[var(--oh-muted)]">
          {t(I18nKey.SECURITY$SAST_TITLE)}
        </p>
        <SecurityScanResults result={lastResult} />
      </div>
    </div>
  );
}
