import type { ReactNode } from "react";
import { Folder } from "lucide-react";
import { FaBitbucket, FaGithub, FaGitlab } from "react-icons/fa6";
import { FaCodeBranch } from "react-icons/fa";
import type { IconType } from "react-icons/lib";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import type { RepositorySelection } from "#/api/open-hands.types";
import type { Provider } from "#/types/settings";
import type { ExecutionStatus } from "#/types/agent-server/core/base/common";
import type { SandboxStatus } from "#/api/conversation-service/agent-server-conversation-service.types";
import AzureDevOpsLogo from "#/assets/branding/azure-devops-logo.svg?react";
import { ConversationStatusDot } from "../conversation-status-dot";

interface ConversationCardPreviewProps {
  title: string;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  selectedRepository: RepositorySelection | null;
  workspaceWorkingDir?: string | null;
  llmModel?: string | null;
  createdAt?: string;
}

const providerIcon: Partial<Record<Provider, IconType>> = {
  bitbucket: FaBitbucket,
  bitbucket_data_center: FaBitbucket,
  github: FaGithub,
  gitlab: FaGitlab,
};

interface PreviewRowProps {
  label: string;
  children: ReactNode;
}

function PreviewRow({ label, children }: PreviewRowProps) {
  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="w-20 shrink-0 text-[var(--oh-muted)]">{label}</span>
      <span className="min-w-0 flex-1 break-words text-[var(--oh-foreground)]">
        {children}
      </span>
    </div>
  );
}

export function ConversationCardPreview({
  title,
  executionStatus,
  sandboxStatus,
  selectedRepository,
  workspaceWorkingDir,
  llmModel,
  createdAt,
}: ConversationCardPreviewProps) {
  const { t } = useTranslation("openhands");

  const repository = selectedRepository?.selected_repository ?? null;
  const branch = selectedRepository?.selected_branch ?? null;
  const provider = selectedRepository?.git_provider ?? null;
  const ProviderIcon = provider ? providerIcon[provider] : null;

  const createdLabel = createdAt
    ? new Date(createdAt).toLocaleString(undefined, {
        dateStyle: "medium",
        timeStyle: "short",
      })
    : null;

  return (
    <div className="flex w-[280px] flex-col gap-3 p-3">
      <div className="flex items-start gap-2">
        {executionStatus !== undefined ? (
          <div className="mt-0.5">
            <ConversationStatusDot
              executionStatus={executionStatus}
              sandboxStatus={sandboxStatus}
              showTooltip={false}
            />
          </div>
        ) : null}
        <span className="break-words text-sm font-medium text-white">
          {title}
        </span>
      </div>

      <dl className="flex flex-col gap-1.5">
        {repository ? (
          <>
            <PreviewRow label={t(I18nKey.CONVERSATION$REPOSITORY)}>
              <span className="inline-flex min-w-0 items-start gap-1.5">
                {ProviderIcon ? (
                  <ProviderIcon size={12} className="mt-0.5 shrink-0" />
                ) : null}
                {provider === "azure_devops" ? (
                  <AzureDevOpsLogo className="mt-0.5 h-3 w-3 shrink-0" />
                ) : null}
                <span className="break-all">{repository}</span>
              </span>
            </PreviewRow>
            {branch ? (
              <PreviewRow label={t(I18nKey.CONVERSATION$BRANCH)}>
                <span className="inline-flex min-w-0 items-start gap-1.5">
                  <FaCodeBranch size={11} className="mt-0.5 shrink-0" />
                  <span className="break-all">{branch}</span>
                </span>
              </PreviewRow>
            ) : null}
          </>
        ) : workspaceWorkingDir ? (
          <PreviewRow label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_DIRECTORY)}>
            <span className="inline-flex min-w-0 items-start gap-1.5">
              <Folder size={12} className="mt-0.5 shrink-0" />
              <span className="break-all">{workspaceWorkingDir}</span>
            </span>
          </PreviewRow>
        ) : null}

        {llmModel ? (
          <PreviewRow label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_MODEL)}>
            <span className="break-all">{llmModel}</span>
          </PreviewRow>
        ) : null}

        {createdLabel ? (
          <PreviewRow label={t(I18nKey.CONVERSATION$CREATED)}>
            {createdLabel}
          </PreviewRow>
        ) : null}
      </dl>
    </div>
  );
}
