import React from "react";
import { useTranslation } from "react-i18next";
import {
  ArrowUpCircle,
  Check,
  CheckCircle2,
  Copy,
  ExternalLink,
  RefreshCw,
} from "lucide-react";
import {
  AGENT_CANVAS_RELEASE_NOTES_URL,
  AGENT_CANVAS_UPDATE_COMMANDS,
} from "#/api/agent-canvas-updates";
import { ModalBackdrop } from "#/components/shared/modals/modal-backdrop";
import { ModalCloseButton } from "#/components/shared/modals/modal-close-button";
import DockerIcon from "#/icons/docker.svg?react";
import NpmIcon from "#/icons/npm.svg?react";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";

const COPY_FEEDBACK_MS = 2000;

type UpdateCommandTab = "npm" | "docker";

const UPDATE_COMMAND_TABS: UpdateCommandTab[] = ["npm", "docker"];

interface AgentCanvasVersionModalProps {
  installedVersion: string;
  latestVersion: string | null;
  updateAvailable: boolean;
  isChecking: boolean;
  onCheckForUpdates: () => void;
  onClose: () => void;
}

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

export function AgentCanvasVersionModal({
  installedVersion,
  latestVersion,
  updateAvailable,
  isChecking,
  onCheckForUpdates,
  onClose,
}: AgentCanvasVersionModalProps) {
  const { t } = useTranslation("openhands");
  const [selectedTab, setSelectedTab] = React.useState<UpdateCommandTab>("npm");
  const [copied, setCopied] = React.useState(false);
  const copiedTimeoutRef = React.useRef<number | null>(null);
  const command =
    selectedTab === "npm"
      ? AGENT_CANVAS_UPDATE_COMMANDS.npm
      : AGENT_CANVAS_UPDATE_COMMANDS.docker;

  React.useEffect(
    () => () => {
      if (copiedTimeoutRef.current !== null) {
        window.clearTimeout(copiedTimeoutRef.current);
      }
    },
    [],
  );

  React.useEffect(() => {
    setCopied(false);
    if (copiedTimeoutRef.current !== null) {
      window.clearTimeout(copiedTimeoutRef.current);
      copiedTimeoutRef.current = null;
    }
  }, [selectedTab]);

  const copyCommand = React.useCallback(() => {
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
    <ModalBackdrop
      onClose={onClose}
      aria-label={t(I18nKey.SETTINGS$VERSION_MODAL_ARIA_LABEL)}
    >
      <section className="relative flex w-[520px] max-w-[90vw] flex-col gap-5 rounded-xl border border-[var(--oh-border)] bg-base-secondary p-6 shadow-xl">
        <ModalCloseButton
          onClose={onClose}
          testId="agent-canvas-version-modal-close"
        />

        <header className="flex items-center gap-3 pr-8">
          {updateAvailable ? (
            <ArrowUpCircle
              className="size-7 shrink-0 text-[#3B82F6]"
              aria-hidden
            />
          ) : (
            <CheckCircle2
              className="size-7 shrink-0 text-[var(--oh-status-success)]"
              aria-hidden
            />
          )}
          <h2 className="text-base font-semibold leading-6 text-white">
            {t(
              updateAvailable
                ? I18nKey.SETTINGS$VERSION_UPDATE_AVAILABLE
                : I18nKey.SETTINGS$VERSION_UP_TO_DATE_TITLE,
            )}
          </h2>
          {updateAvailable && latestVersion ? (
            <span className="rounded-full bg-[#1E3A5F] px-2 py-0.5 text-xs font-semibold text-[#3B82F6]">
              {latestVersion}
            </span>
          ) : null}
        </header>

        <div className="flex flex-col gap-1">
          {updateAvailable && latestVersion ? (
            <p className="text-sm text-[var(--oh-text)]">
              {t(I18nKey.SETTINGS$VERSION_UPDATE_MESSAGE, {
                installed: installedVersion,
                latest: latestVersion,
              })}{" "}
              <a
                href={AGENT_CANVAS_RELEASE_NOTES_URL}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-1.5 whitespace-nowrap font-medium text-white hover:text-[var(--oh-text)]"
              >
                {t(I18nKey.SETTINGS$VERSION_RELEASE_NOTES)}
                <ExternalLink className="size-4 shrink-0" aria-hidden />
              </a>
            </p>
          ) : (
            <>
              <p className="text-sm text-[var(--oh-text)]">
                {t(I18nKey.SETTINGS$VERSION_INSTALLED, {
                  version: installedVersion,
                })}
              </p>
              {latestVersion ? (
                <p className="whitespace-nowrap text-sm text-[var(--oh-status-success)]">
                  {t(I18nKey.SETTINGS$VERSION_LATEST_MESSAGE)}{" "}
                  <a
                    href={AGENT_CANVAS_RELEASE_NOTES_URL}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-1.5 font-medium text-white hover:text-[var(--oh-text)]"
                  >
                    {t(I18nKey.SETTINGS$VERSION_RELEASE_NOTES)}
                    <ExternalLink className="size-4 shrink-0" aria-hidden />
                  </a>
                </p>
              ) : (
                <p className="text-sm text-[var(--oh-muted)]">
                  {t(I18nKey.SETTINGS$VERSION_CHECK_UNAVAILABLE)}
                </p>
              )}
            </>
          )}
        </div>

        {updateAvailable ? (
          <div>
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
              <code className="min-w-0 flex-1 overflow-visible whitespace-nowrap font-mono text-sm text-white">
                {command}
              </code>
              <button
                type="button"
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
        ) : null}

        {!updateAvailable ? (
          <div className="flex flex-wrap gap-x-8 gap-y-3">
            <button
              type="button"
              onClick={onCheckForUpdates}
              className="inline-flex items-center gap-2 text-sm font-medium text-white hover:text-[var(--oh-text)] disabled:cursor-wait disabled:text-[var(--oh-muted)]"
              disabled={isChecking}
            >
              {t(I18nKey.SETTINGS$VERSION_CHECK_FOR_UPDATES)}
              <RefreshCw
                className={cn("size-4 shrink-0", isChecking && "animate-spin")}
                aria-hidden
              />
            </button>
          </div>
        ) : null}
      </section>
    </ModalBackdrop>
  );
}
