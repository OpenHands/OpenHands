import { useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import {
  ArrowUpCircle,
  Check,
  CircleCheck,
  Copy,
  ExternalLink,
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
import { BaseModalTitle } from "#/components/shared/modals/confirmation-modals/base-modal";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalBody } from "#/components/shared/modals/modal-body";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import { useLatestAgentCanvasVersion } from "#/hooks/query/use-latest-agent-canvas-version";
import DockerIcon from "#/icons/docker.svg?react";
import NpmIcon from "#/icons/npm.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const COPY_FEEDBACK_MS = 2000;

type UpdateCommandTab = "npm" | "docker";

const UPDATE_COMMAND_TABS: UpdateCommandTab[] = ["npm", "docker"];

function getUpdateCommandTabLabelKey(tab: UpdateCommandTab): I18nKey {
  return tab === "npm"
    ? I18nKey.SETTINGS$VERSION_NPM_RECOMMENDED
    : I18nKey.SETTINGS$VERSION_DOCKER;
}

function UpdateCommandTabIcon({ tab }: { tab: UpdateCommandTab }) {
  if (tab === "npm") {
    return <NpmIcon className="size-4 shrink-0" aria-hidden />;
  }
  return <DockerIcon className="size-5 shrink-0" aria-hidden />;
}

function UpdateCommandTabs() {
  const { t } = useTranslation("openhands");
  const [selectedTab, setSelectedTab] = useState<UpdateCommandTab>("npm");
  const [copied, setCopied] = useState(false);
  const copiedTimeoutRef = useRef<number | null>(null);
  const command =
    selectedTab === "npm"
      ? AGENT_CANVAS_UPDATE_COMMANDS.npm
      : AGENT_CANVAS_UPDATE_COMMANDS.docker;

  useEffect(
    () => () => {
      if (copiedTimeoutRef.current !== null) {
        window.clearTimeout(copiedTimeoutRef.current);
      }
    },
    [],
  );

  useEffect(() => {
    setCopied(false);
    if (copiedTimeoutRef.current !== null) {
      window.clearTimeout(copiedTimeoutRef.current);
      copiedTimeoutRef.current = null;
    }
  }, [selectedTab]);

  const copyCommand = useCallback(() => {
    void navigator.clipboard?.writeText(command);
    setCopied(true);
    if (copiedTimeoutRef.current !== null) {
      window.clearTimeout(copiedTimeoutRef.current);
    }
    copiedTimeoutRef.current = window.setTimeout(() => {
      setCopied(false);
      copiedTimeoutRef.current = null;
    }, COPY_FEEDBACK_MS);
  }, [command]);

  return (
    <div className="w-full">
      <div className="flex justify-center">
        {UPDATE_COMMAND_TABS.map((tab) => (
          <button
            key={tab}
            type="button"
            onClick={() => setSelectedTab(tab)}
            className={cn(
              "inline-flex items-center gap-1.5 px-3 pb-2 text-sm font-medium",
              selectedTab === tab
                ? "border-b border-white text-white"
                : "text-[var(--oh-muted)] hover:text-white",
            )}
          >
            <UpdateCommandTabIcon tab={tab} />
            {t(getUpdateCommandTabLabelKey(tab))}
          </button>
        ))}
      </div>
      <div className="flex items-center gap-3 rounded-lg border border-[var(--oh-border)] bg-[var(--oh-surface-deep)] px-4 py-3">
        <code
          data-testid={
            selectedTab === "npm"
              ? "agent-canvas-update-command-npm"
              : "agent-canvas-update-command-docker"
          }
          className="min-w-0 flex-1 overflow-visible whitespace-nowrap font-mono text-sm text-white"
        >
          {command}
        </code>
        <button
          type="button"
          data-testid="copy-to-clipboard"
          onClick={copyCommand}
          aria-label={t(
            copied
              ? I18nKey.BUTTON$COPIED
              : I18nKey.SETTINGS$VERSION_COPY_COMMAND,
          )}
          disabled={copied}
          className="shrink-0 text-[var(--oh-muted)] hover:text-white disabled:hover:text-[var(--oh-muted)]"
        >
          {copied ? (
            <Check
              className="size-4 text-[var(--oh-status-success)]"
              aria-hidden
            />
          ) : (
            <Copy className="size-4" aria-hidden />
          )}
        </button>
      </div>
    </div>
  );
}

interface AgentCanvasUpdateModalProps {
  onClose: () => void;
  latestVersion: string | undefined;
  isPending: boolean;
  isFetching: boolean;
  updateAvailable: boolean;
  upToDate: boolean;
  onCheckForUpdates: () => void;
}

function AgentCanvasUpdateModal({
  onClose,
  latestVersion,
  isPending,
  isFetching,
  updateAvailable,
  upToDate,
  onCheckForUpdates,
}: AgentCanvasUpdateModalProps) {
  const { t } = useTranslation("openhands");

  return (
    <ModalBackdrop onClose={onClose}>
      <ModalBody
        width="md"
        className="relative flex max-h-[80vh] flex-col items-start overflow-auto border border-[var(--oh-border)]"
        testID="agent-canvas-update-modal"
      >
        <ModalCloseButton
          onClose={onClose}
          testId="close-agent-canvas-update-modal"
        />
        <div className="flex w-full flex-col gap-2 pr-10">
          <BaseModalTitle title={t(I18nKey.SETTINGS$APP_UPDATE_CARD_TITLE)} />
          <div className="flex items-center gap-1.5 text-xs">
            <span className="text-[var(--oh-text-dim)]">
              {t(I18nKey.SETTINGS$APP_UPDATE_VERSION_LABEL)}
            </span>
            <span className="text-white">{AGENT_CANVAS_CLIENT_VERSION}</span>
          </div>
        </div>

        <div className="flex w-full flex-col gap-3">
          <BrandButton
            testId="agent-canvas-update-check-button"
            type="button"
            variant="secondary"
            className="w-full"
            isDisabled={isFetching}
            aria-busy={isFetching}
            onClick={onCheckForUpdates}
            startContent={
              <RefreshCw
                className={cn("size-4", isFetching && "animate-spin")}
                aria-hidden
              />
            }
          >
            {t(I18nKey.SETTINGS$APP_UPDATE_CHECK_BUTTON)}
          </BrandButton>

          <div data-testid="agent-canvas-update-status" className="w-full">
            {isPending || isFetching ? (
              <span className="text-xs text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$APP_UPDATE_CHECKING)}
              </span>
            ) : updateAvailable ? (
              <div className="flex items-start gap-2 rounded-lg border border-[#3B82F6]/30 bg-[#1E3A5F] px-3 py-2 text-xs text-[#3B82F6]">
                <ArrowUpCircle className="size-4 shrink-0" aria-hidden />
                <span>
                  {t(I18nKey.SETTINGS$APP_UPDATE_AVAILABLE_MESSAGE, {
                    version: latestVersion,
                  })}{" "}
                  <a
                    data-testid="agent-canvas-update-release-notes"
                    href={AGENT_CANVAS_RELEASE_NOTES_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 whitespace-nowrap font-medium text-white hover:text-[var(--oh-text)]"
                  >
                    {t(I18nKey.SETTINGS$VERSION_RELEASE_NOTES)}
                    <ExternalLink className="size-3.5 shrink-0" aria-hidden />
                  </a>
                </span>
              </div>
            ) : upToDate ? (
              <div className="flex items-start gap-2 rounded-lg border border-[var(--oh-status-success)]/30 bg-[var(--oh-status-success)]/10 px-3 py-2 text-xs text-[var(--oh-status-success)]">
                <CircleCheck className="size-4 shrink-0" aria-hidden />
                <span>
                  {t(I18nKey.SETTINGS$APP_UPDATE_LATEST_MESSAGE)}{" "}
                  <a
                    data-testid="agent-canvas-update-release-notes"
                    href={AGENT_CANVAS_RELEASE_NOTES_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 whitespace-nowrap font-medium text-white hover:text-[var(--oh-text)]"
                  >
                    {t(I18nKey.SETTINGS$VERSION_RELEASE_NOTES)}
                    <ExternalLink className="size-3.5 shrink-0" aria-hidden />
                  </a>
                </span>
              </div>
            ) : (
              <span className="text-xs text-[var(--oh-muted)]">
                {t(I18nKey.SETTINGS$APP_UPDATE_CHECK_FAILED)}
              </span>
            )}
          </div>

          {updateAvailable ? (
            <div className="flex w-full flex-col border-t border-[var(--oh-border)] pt-3">
              <span className="text-sm font-medium text-white">
                {t(I18nKey.SETTINGS$APP_UPDATE_HOW_TO_UPDATE)}
              </span>
              <span className="mt-1 mb-4 text-sm text-[var(--oh-text-dim)]">
                {t(I18nKey.SETTINGS$APP_UPDATE_RUN_COMMANDS)}
              </span>
              <UpdateCommandTabs />
            </div>
          ) : null}
        </div>
      </ModalBody>
    </ModalBackdrop>
  );
}

interface AgentCanvasUpdateCardProps {
  /** When true, omit the card unless an update is available (main sidebar). */
  hideWhenUpToDate?: boolean;
}

/**
 * Information-only version/update card for settings and the main sidebar.
 * Opens update details in a modal. Never blocks or toasts; check failures
 * degrade to a quiet inline message. Hidden in locked-to-Cloud deployments —
 * the hosted canvas has no npm/docker install to update.
 */
export function AgentCanvasUpdateCard({
  hideWhenUpToDate = false,
}: AgentCanvasUpdateCardProps) {
  const { t } = useTranslation("openhands");
  const [isModalOpen, setIsModalOpen] = useState(false);
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

  if (hideWhenUpToDate && !updateAvailable) return null;

  return (
    <>
      <section data-testid="agent-canvas-update-card">
        <button
          type="button"
          onClick={() => setIsModalOpen(true)}
          aria-haspopup="dialog"
          data-testid="agent-canvas-update-toggle"
          className="flex w-full cursor-pointer flex-col gap-1 rounded-md border border-[var(--oh-border)] bg-base-secondary px-3 py-2 text-left hover:bg-[var(--oh-surface-raised)]"
        >
          <span className="flex w-full items-center gap-2">
            <span className="flex-1 truncate text-sm font-semibold leading-5 text-white">
              {t(I18nKey.SETTINGS$APP_UPDATE_CARD_TITLE)}
            </span>
            {comparison !== null && (
              <span
                data-testid="agent-canvas-update-badge"
                className={cn(
                  "inline-flex shrink-0 items-center gap-1 text-[10px] font-medium leading-none",
                  updateAvailable
                    ? "rounded-full border border-transparent bg-[#1E3A5F] px-1.5 py-0.5 text-[#3B82F6]"
                    : "text-success",
                )}
              >
                {updateAvailable ? (
                  <ArrowUpCircle className="size-3 shrink-0" aria-hidden />
                ) : (
                  <CircleCheck className="size-3 shrink-0" aria-hidden />
                )}
                {t(
                  updateAvailable
                    ? I18nKey.SETTINGS$APP_UPDATE_BADGE_UPDATE_AVAILABLE
                    : I18nKey.SETTINGS$APP_UPDATE_BADGE_UP_TO_DATE,
                )}
              </span>
            )}
          </span>

          <span className="flex items-center gap-1.5 text-xs leading-5 text-[var(--oh-muted)]">
            <span>{t(I18nKey.SETTINGS$APP_UPDATE_VERSION_LABEL)}</span>
            <span className="text-white">{AGENT_CANVAS_CLIENT_VERSION}</span>
          </span>
        </button>
      </section>

      {isModalOpen && (
        <AgentCanvasUpdateModal
          onClose={() => setIsModalOpen(false)}
          latestVersion={latestVersion}
          isPending={latestVersionQuery.isPending}
          isFetching={latestVersionQuery.isFetching}
          updateAvailable={updateAvailable}
          upToDate={upToDate}
          onCheckForUpdates={() => latestVersionQuery.refetch()}
        />
      )}
    </>
  );
}
