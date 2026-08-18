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
import { getDisplayConversationTags } from "#/api/agent-server-adapter";
import { resolveAcpProviderIcon } from "#/constants/acp-providers";
import AzureDevOpsLogo from "#/assets/branding/azure-devops-logo.svg?react";
import { AgentBrandIcon } from "#/components/shared/agent-brand-icon";
import { ConversationStatusDot } from "../conversation-status-dot";
import { getConversationTagLabel } from "./conversation-tag-display";
import { getConversationTagIcon } from "./conversation-tag-icons";

interface ConversationCardPreviewProps {
  title: string;
  executionStatus?: ExecutionStatus | null;
  sandboxStatus?: SandboxStatus | null;
  selectedRepository: RepositorySelection | null;
  workspaceWorkingDir?: string | null;
  llmModel?: string | null;
  /**
   * High-level kind of the conversation's agent. Drives the model row's brand
   * mark exactly like the card chip does — without it an ACP conversation
   * would show the OpenHands wordmark next to a Claude Code / Codex / Gemini
   * model, contradicting the chip on the very card being hovered.
   */
  agentKind?: "openhands" | "acp" | null;
  /** Registry key of the ACP CLI server, resolved to its brand mark. */
  acpServer?: string | null;
  createdAt?: string;
  /**
   * Server-side conversation tags. Always shown in the hovercard when present
   * (except keys already covered by repository / branch / directory rows).
   * Sidebar card chips stay gated by the panel's Tags preference.
   */
  tags?: Record<string, string> | null;
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
    <div className="flex items-start gap-2 text-xs leading-4">
      <span className="w-20 shrink-0 whitespace-normal break-words text-[var(--oh-muted)]">
        {label}
      </span>
      <span className="min-w-0 flex-1 overflow-visible whitespace-normal break-words text-[var(--oh-foreground)]">
        {children}
      </span>
    </div>
  );
}

/**
 * Icon + value for preview rows. The icon sits in a box matching the first
 * line's height (``leading-4``) so it stays optically centered when the value
 * wraps to multiple lines. The slot has a 12px *minimum* rather than a fixed
 * width so the wider OpenHands wordmark (18px at ``size={12}``) is not clipped,
 * while the square icons still share one column.
 */
function PreviewValueWithIcon({
  icon,
  children,
  testId,
  tagKey,
}: {
  icon?: ReactNode;
  children: ReactNode;
  testId?: string;
  tagKey?: string;
}) {
  return (
    <span
      data-testid={testId}
      data-tag-key={tagKey}
      className="inline-flex min-w-0 max-w-full items-start gap-1.5"
    >
      {icon ? (
        <span className="inline-flex h-4 min-w-3 shrink-0 items-center justify-center [&_svg]:block">
          {icon}
        </span>
      ) : null}
      <span className="min-w-0 flex-1 whitespace-normal break-all">
        {children}
      </span>
    </span>
  );
}

export function ConversationCardPreview({
  title,
  executionStatus,
  sandboxStatus,
  selectedRepository,
  workspaceWorkingDir,
  llmModel,
  agentKind = null,
  acpServer = null,
  createdAt,
  tags = null,
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

  const previewTags = getDisplayConversationTags(tags);

  return (
    <div
      data-testid="conversation-card-preview"
      className="flex w-[280px] max-w-[min(280px,90vw)] flex-col gap-3 overflow-visible p-3"
    >
      <div className="flex items-start gap-2">
        {executionStatus !== undefined ? (
          <span className="inline-flex h-5 w-2.5 shrink-0 items-center justify-center">
            <ConversationStatusDot
              executionStatus={executionStatus}
              sandboxStatus={sandboxStatus}
              showTooltip={false}
            />
          </span>
        ) : null}
        <span className="break-words text-sm font-medium leading-5 text-white">
          {title}
        </span>
      </div>

      <dl className="flex flex-col gap-1.5">
        {repository ? (
          <>
            <PreviewRow label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_REPO)}>
              <PreviewValueWithIcon
                icon={
                  ProviderIcon ? (
                    <ProviderIcon size={12} />
                  ) : provider === "azure_devops" ? (
                    <AzureDevOpsLogo className="h-3 w-3" />
                  ) : undefined
                }
              >
                {repository}
              </PreviewValueWithIcon>
            </PreviewRow>
            {branch ? (
              <PreviewRow label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_BRANCH)}>
                <PreviewValueWithIcon icon={<FaCodeBranch size={11} />}>
                  {branch}
                </PreviewValueWithIcon>
              </PreviewRow>
            ) : null}
          </>
        ) : workspaceWorkingDir ? (
          <PreviewRow label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_DIRECTORY)}>
            <PreviewValueWithIcon icon={<Folder size={12} />}>
              {workspaceWorkingDir}
            </PreviewValueWithIcon>
          </PreviewRow>
        ) : null}

        {llmModel ? (
          <PreviewRow label={t(I18nKey.CONVERSATION_PANEL$PREVIEW_MODEL)}>
            <PreviewValueWithIcon
              testId="conversation-card-preview-model"
              icon={
                <AgentBrandIcon
                  kind={
                    agentKind === "acp"
                      ? resolveAcpProviderIcon(acpServer)
                      : "openhands"
                  }
                  size={12}
                />
              }
            >
              {llmModel}
            </PreviewValueWithIcon>
          </PreviewRow>
        ) : null}

        {previewTags.map(([key, value]) => {
          const Icon = getConversationTagIcon(key, value);
          const label = getConversationTagLabel(key, t);

          return (
            <PreviewRow key={key} label={label}>
              <PreviewValueWithIcon
                testId="conversation-card-preview-tag-row"
                tagKey={key}
                icon={<Icon aria-hidden className="h-3 w-3" />}
              >
                {value}
              </PreviewValueWithIcon>
            </PreviewRow>
          );
        })}

        {createdLabel ? (
          <PreviewRow label={t(I18nKey.CONVERSATION$CREATED)}>
            {createdLabel}
          </PreviewRow>
        ) : null}
      </dl>
    </div>
  );
}
