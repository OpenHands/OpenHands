import { useId, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ChevronDown,
  ChevronUp,
  CircleCheck,
  ExternalLink,
  Info,
  RefreshCw,
} from "lucide-react";
import {
  AGENT_CANVAS_RELEASE_NOTES_URL,
  AGENT_CANVAS_UPDATE_COMMANDS,
} from "#/api/agent-canvas-updates";
import { compareAgentServerVersions } from "#/api/agent-server-compatibility";
import { getLockedCloudHost } from "#/api/agent-server-config";
import { AGENT_CANVAS_CLIENT_VERSION } from "#/api/client-source";
import { BrandButton } from "#/components/features/settings/brand-button";
import { CopyToClipboardButton } from "#/components/shared/buttons/copy-to-clipboard-button";
import { useLatestAgentCanvasVersion } from "#/hooks/query/use-latest-agent-canvas-version";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

function UpdateCommandRow({
  command,
  testId,
}: {
  command: string;
  testId: string;
}) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(command);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex items-center gap-2 rounded-lg bg-[var(--oh-surface-deep)] px-2 py-1.5">
      <code
        data-testid={testId}
        className="min-w-0 flex-1 select-all break-all font-mono text-[11px] leading-4 text-white"
      >
        {command}
      </code>
      <CopyToClipboardButton
        isHidden={false}
        isDisabled={false}
        onClick={handleCopy}
        mode={copied ? "copied" : "copy"}
      />
    </div>
  );
}

/**
 * Information-only version/update card for the settings navigation surfaces.
 * Never blocks or toasts; check failures degrade to a quiet inline message.
 * Hidden in locked-to-Cloud deployments — the hosted canvas has no npm/docker
 * install to update.
 */
export function AgentCanvasUpdateCard() {
  const { t } = useTranslation("openhands");
  const [expanded, setExpanded] = useState(false);
  // The card is mounted on multiple settings surfaces at once; ids must not
  // collide across instances.
  const detailsId = useId();
  const isLockedToCloud = getLockedCloudHost() !== null;
  const latestVersionQuery = useLatestAgentCanvasVersion({
    enabled: !isLockedToCloud,
  });

  if (isLockedToCloud) return null;

  const latestVersion = latestVersionQuery.data;
  const comparison =
    latestVersion !== undefined
      ? compareAgentServerVersions(latestVersion, AGENT_CANVAS_CLIENT_VERSION)
      : null;
  const updateAvailable = comparison === 1;
  // 0 or -1: running the npm latest or newer (local dev build).
  const upToDate = comparison !== null && comparison <= 0;
  const ChevronIcon = expanded ? ChevronUp : ChevronDown;

  return (
    <section
      data-testid="agent-canvas-update-card"
      className="flex flex-col gap-2 rounded-xl border border-[var(--oh-border)] bg-[var(--oh-surface-raised)] p-3"
    >
      <button
        type="button"
        onClick={() => setExpanded((prev) => !prev)}
        aria-expanded={expanded}
        aria-controls={detailsId}
        data-testid="agent-canvas-update-toggle"
        className="flex w-full cursor-pointer items-center gap-2 text-left"
      >
        <span className="flex-1 truncate text-xs font-medium text-white">
          {t(I18nKey.SETTINGS$APP_UPDATE_CARD_TITLE)}
        </span>
        {comparison !== null && (
          <span
            data-testid="agent-canvas-update-badge"
            className={cn(
              "inline-flex shrink-0 items-center rounded-full border border-[var(--oh-border)] bg-[var(--oh-surface)] px-1.5 py-0.5 text-[10px] font-medium leading-none",
              updateAvailable ? "text-warning" : "text-success",
            )}
          >
            {t(
              updateAvailable
                ? I18nKey.SETTINGS$APP_UPDATE_BADGE_UPDATE_AVAILABLE
                : I18nKey.SETTINGS$APP_UPDATE_BADGE_UP_TO_DATE,
            )}
          </span>
        )}
        <ChevronIcon
          className="size-4 shrink-0 text-[var(--oh-muted)]"
          aria-hidden
        />
      </button>

      <div className="flex items-center justify-between text-xs">
        <span className="text-[var(--oh-text-dim)]">
          {t(I18nKey.SETTINGS$APP_UPDATE_VERSION_LABEL)}
        </span>
        <span className="font-mono text-white">
          {AGENT_CANVAS_CLIENT_VERSION}
        </span>
      </div>

      {expanded && (
        <div id={detailsId} className="flex flex-col gap-3 pt-1">
          <div
            data-testid="agent-canvas-update-status"
            className="flex items-start gap-2 text-xs"
          >
            {latestVersionQuery.isPending ? (
              <span className="text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$APP_UPDATE_CHECKING)}
              </span>
            ) : updateAvailable ? (
              <>
                <Info className="size-4 shrink-0 text-warning" aria-hidden />
                <span className="text-warning">
                  {t(I18nKey.SETTINGS$APP_UPDATE_AVAILABLE_MESSAGE, {
                    version: latestVersion,
                  })}
                </span>
              </>
            ) : upToDate ? (
              <>
                <CircleCheck
                  className="size-4 shrink-0 text-success"
                  aria-hidden
                />
                <span className="text-success">
                  {t(I18nKey.SETTINGS$APP_UPDATE_LATEST_MESSAGE)}
                </span>
              </>
            ) : (
              <span className="text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$APP_UPDATE_CHECK_FAILED)}
              </span>
            )}
          </div>

          <BrandButton
            testId="agent-canvas-update-check-button"
            type="button"
            variant="secondary"
            className="w-full"
            isDisabled={latestVersionQuery.isFetching}
            aria-busy={latestVersionQuery.isFetching}
            onClick={() => latestVersionQuery.refetch()}
            startContent={
              <RefreshCw
                className={cn(
                  "size-4",
                  latestVersionQuery.isFetching && "animate-spin",
                )}
                aria-hidden
              />
            }
          >
            {t(I18nKey.SETTINGS$APP_UPDATE_CHECK_BUTTON)}
          </BrandButton>

          <div className="flex flex-col gap-2 rounded-lg border border-[var(--oh-border-subtle)] p-3 text-xs">
            <div className="flex items-center gap-1.5 text-white">
              <Info className="size-4 shrink-0" aria-hidden />
              <span className="font-medium">
                {t(I18nKey.SETTINGS$APP_UPDATE_HOW_TO_UPDATE)}
              </span>
            </div>
            <span className="text-[var(--oh-text-dim)]">
              {t(I18nKey.SETTINGS$APP_UPDATE_RUN_COMMANDS)}
            </span>
            <UpdateCommandRow
              command={AGENT_CANVAS_UPDATE_COMMANDS.npm}
              testId="agent-canvas-update-command-npm"
            />
            <span className="text-center text-[var(--oh-text-dim)]">
              {t(I18nKey.SETTINGS$APP_UPDATE_OR)}
            </span>
            <UpdateCommandRow
              command={AGENT_CANVAS_UPDATE_COMMANDS.docker}
              testId="agent-canvas-update-command-docker"
            />
          </div>

          <a
            data-testid="agent-canvas-update-release-notes"
            href={AGENT_CANVAS_RELEASE_NOTES_URL}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1 self-start text-xs text-[var(--oh-accent)] underline"
          >
            {t(I18nKey.SETTINGS$APP_UPDATE_RELEASE_NOTES)}
            <ExternalLink size={12} aria-hidden />
          </a>
        </div>
      )}
    </section>
  );
}
