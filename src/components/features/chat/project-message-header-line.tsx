import { CornerDownRight } from "lucide-react";
import { useTranslation } from "react-i18next";
import { I18nKey } from "#/i18n/declaration";
import { ConversationMention } from "#/components/features/markdown/conversation-mention";
import type {
  ProjectHeaderKind,
  ProjectMessageHeader,
} from "#/utils/project-message-header";

const LABELS: Record<ProjectHeaderKind, I18nKey> = {
  sent_by: I18nKey.PROJECTS$HEADER_SENT_BY,
  update_from: I18nKey.PROJECTS$HEADER_UPDATE_FROM,
  reply_in: I18nKey.PROJECTS$HEADER_REPLY_IN,
  person_in: I18nKey.PROJECTS$HEADER_PERSON_IN,
};

/** One compact line in place of the Agent Server's header for the agent. */
export function ProjectMessageHeaderLine({
  header,
}: {
  header: ProjectMessageHeader;
}) {
  const { t } = useTranslation("openhands");
  const title = header.title || header.conversationId?.slice(0, 8) || "";
  return (
    <div
      data-testid="project-message-header"
      data-kind={header.kind}
      className="flex items-center gap-1.5 text-xs text-muted"
    >
      <CornerDownRight size={12} aria-hidden="true" />
      <span>{t(LABELS[header.kind])}</span>
      {header.conversationId ? (
        <ConversationMention id={header.conversationId}>
          {title}
        </ConversationMention>
      ) : (
        <span>{title}</span>
      )}
    </div>
  );
}
