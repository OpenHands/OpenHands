import { useTranslation } from "react-i18next";
import type { MobileArtifact } from "#/api/pentest/mobile-artifacts-types";
import { I18nKey } from "#/i18n/declaration";

type EmulatorArtifactsListProps = {
  artifacts: MobileArtifact[];
  onRequestInstall?: (artifactId: string) => void;
  installEnabled?: boolean;
};

function scanLabelKey(status: MobileArtifact["scan_status"]) {
  switch (status) {
    case "queued":
      return I18nKey.EMULATOR$SCAN_QUEUED;
    case "scanning":
      return I18nKey.EMULATOR$SCAN_SCANNING;
    case "ready":
      return I18nKey.EMULATOR$SCAN_READY;
    case "failed":
      return I18nKey.EMULATOR$SCAN_FAILED;
    default:
      return I18nKey.EMULATOR$SCAN_QUEUED;
  }
}

export function EmulatorArtifactsList({
  artifacts,
  onRequestInstall,
  installEnabled = false,
}: EmulatorArtifactsListProps) {
  const { t } = useTranslation("openhands");

  if (artifacts.length === 0) {
    return (
      <p
        className="px-1 text-xs text-[var(--oh-muted)]"
        data-testid="emulator-artifacts-empty"
      >
        {t(I18nKey.EMULATOR$ARTIFACTS_EMPTY)}
      </p>
    );
  }

  return (
    <ul
      className="flex max-h-24 flex-col gap-1 overflow-y-auto"
      data-testid="emulator-artifacts-list"
    >
      {artifacts.map((artifact) => (
        <li
          key={artifact.artifact_id}
          className="flex items-center justify-between gap-2 px-1 text-xs"
          data-artifact-id={artifact.artifact_id}
        >
          <span className="min-w-0 truncate text-[var(--foreground)]">
            {artifact.filename}
          </span>
          <span
            className="shrink-0 text-[var(--oh-muted)]"
            data-testid="emulator-scan-status"
          >
            {t(scanLabelKey(artifact.scan_status))}
          </span>
          {onRequestInstall && artifact.scan_status === "ready" && (
            <button
              type="button"
              className="shrink-0 text-[var(--foreground)] underline disabled:cursor-not-allowed disabled:opacity-50 disabled:no-underline"
              disabled={!installEnabled}
              title={
                installEnabled
                  ? undefined
                  : t(I18nKey.EMULATOR$INSTALL_DISABLED_HINT)
              }
              onClick={() => onRequestInstall(artifact.artifact_id)}
            >
              {t(I18nKey.EMULATOR$INSTALL_TOGGLE)}
            </button>
          )}
        </li>
      ))}
    </ul>
  );
}
