import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { RefreshCw } from "lucide-react";
import type { GitSyncStatus } from "#/types/git-sync";
import { formatTimeDelta } from "#/utils/format-time-delta";
import { parseGitRemoteUrl } from "#/utils/parse-git-remote-url";
import { cn } from "#/utils/utils";
import GitBranchIcon from "#/icons/git-branch.svg?react";
import GlobeIcon from "#/icons/globe.svg?react";
import FolderIcon from "#/icons/folder.svg?react";
import ClockIcon from "#/icons/clock.svg?react";
import { SectionCard } from "#/components/features/automations/detail/section-card";
import { ConfigField } from "#/components/features/automations/detail/config-field";
import { BrandButton } from "#/components/features/settings/brand-button";
import { GitSyncStatusPill } from "./git-sync-status-pill";
import {
  GitSyncActivityRow,
  type GitSyncActivityState,
} from "./git-sync-activity-row";

/**
 * A browsable URL for the configured remote, or `null` when there isn't one.
 * A bare repo on disk is a valid sync target, so a local path -- and anything
 * else that doesn't parse -- stays plain text rather than becoming a dead link.
 *
 * An http(s) remote is already browsable, so it is reused as-is apart from the
 * `.git` suffix and any embedded credentials, which keeps forge-specific paths
 * (Azure's `org/project/_git/repo`) and non-default ports intact. Other schemes
 * -- `ssh://`, `git://`, `git@host:owner/repo` -- carry no browsable form, so
 * they are rebuilt over https from the parsed host and repository.
 */
function browseUrlFor(repoUrl: string): string | null {
  const parsed = parseGitRemoteUrl(repoUrl);
  if (!parsed?.host || !parsed.repository) return null;

  if (/^https?:\/\//i.test(parsed.url)) {
    const url = new URL(parsed.url);
    url.username = "";
    url.password = "";
    url.pathname = url.pathname.replace(/\.git$/, "");
    return url.toString();
  }

  return `https://${parsed.host}/${parsed.repository}`;
}

interface GitSyncOverviewSectionProps {
  status: GitSyncStatus;
  onSyncNow: () => void;
  isSyncing: boolean;
  syncActivity: GitSyncActivityState;
  syncStartedAt: string | null;
  canManage: boolean;
}

export function GitSyncOverviewSection({
  status,
  onSyncNow,
  isSyncing,
  syncActivity,
  syncStartedAt,
  canManage,
}: GitSyncOverviewSectionProps) {
  const { t } = useTranslation("openhands");
  const repoHref = browseUrlFor(status.repo_url);

  return (
    <SectionCard
      icon={<GitBranchIcon className="size-4" />}
      title={t(I18nKey.AUTOMATIONS$GIT_SYNC$STATUS_TITLE)}
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <GitSyncStatusPill
            testId="git-sync-enabled-pill"
            tone={status.enabled ? "success" : "neutral"}
            label={t(
              status.enabled
                ? I18nKey.AUTOMATIONS$GIT_SYNC$ENABLED
                : I18nKey.AUTOMATIONS$GIT_SYNC$DISABLED,
            )}
          />
          <GitSyncStatusPill
            testId="git-sync-encryption-pill"
            tone={status.encryption_enabled ? "success" : "neutral"}
            label={t(
              status.encryption_enabled
                ? I18nKey.AUTOMATIONS$GIT_SYNC$ENCRYPTED
                : I18nKey.AUTOMATIONS$GIT_SYNC$NOT_ENCRYPTED,
            )}
          />
        </div>
        <BrandButton
          testId="git-sync-now-button"
          type="button"
          variant="secondary"
          isDisabled={!canManage || !status.enabled || isSyncing}
          onClick={onSyncNow}
          startContent={
            <RefreshCw
              className={cn("size-4", isSyncing && "animate-spin")}
              aria-hidden
            />
          }
        >
          {t(
            isSyncing
              ? I18nKey.AUTOMATIONS$GIT_SYNC$SYNCING
              : I18nKey.AUTOMATIONS$GIT_SYNC$SYNC_NOW,
          )}
        </BrandButton>
      </div>

      <GitSyncActivityRow
        state={syncActivity}
        startedAt={syncStartedAt}
        pendingCount={status.dirty_count}
      />

      <div className="mt-5 grid grid-cols-2 gap-x-4 gap-y-5">
        <ConfigField
          icon={<GlobeIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$REPOSITORY)}
        >
          {repoHref ? (
            <a
              data-testid="git-sync-repo-link"
              href={repoHref}
              target="_blank"
              rel="noopener noreferrer"
              className="break-all underline transition-colors hover:text-foreground"
            >
              {status.repo_url}
            </a>
          ) : (
            <span className="break-all">
              {status.repo_url ||
                t(I18nKey.AUTOMATIONS$GIT_SYNC$NOT_CONFIGURED)}
            </span>
          )}
        </ConfigField>

        <ConfigField
          icon={<GitBranchIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_BRANCH)}
        >
          {status.branch}
        </ConfigField>

        <ConfigField
          icon={<FolderIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_PATH)}
        >
          {status.path}
        </ConfigField>

        <ConfigField
          icon={<GitBranchIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$LAST_SYNCED_COMMIT)}
        >
          {status.last_synced_commit ? (
            <span className="font-mono">
              {status.last_synced_commit.slice(0, 7)}
            </span>
          ) : (
            t(I18nKey.AUTOMATIONS$GIT_SYNC$NEVER_SYNCED)
          )}
        </ConfigField>

        <ConfigField
          icon={<ClockIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$LAST_SYNCED_AT)}
        >
          {status.last_synced_at
            ? `${formatTimeDelta(status.last_synced_at)} ${t(I18nKey.CONVERSATION$AGO)}`
            : t(I18nKey.AUTOMATIONS$GIT_SYNC$NEVER_SYNCED)}
        </ConfigField>

        <ConfigField
          icon={<ClockIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$FIELD_INTERVAL)}
        >
          {status.interval_seconds > 0
            ? t(I18nKey.AUTOMATIONS$GIT_SYNC$EVERY_N_SECONDS, {
                count: status.interval_seconds,
              })
            : t(I18nKey.AUTOMATIONS$GIT_SYNC$MANUAL_ONLY)}
        </ConfigField>

        <ConfigField
          icon={<ClockIcon className="size-3.5" />}
          label={t(I18nKey.AUTOMATIONS$GIT_SYNC$PENDING_CHANGES)}
        >
          <span
            className={cn(status.dirty_count > 0 && "text-[var(--oh-warning)]")}
          >
            {status.dirty_count}
          </span>
        </ConfigField>
      </div>
    </SectionCard>
  );
}
