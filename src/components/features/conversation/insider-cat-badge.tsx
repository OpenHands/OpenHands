import { Cat } from "lucide-react";
import { useTranslation } from "react-i18next";
import { NavigationLink } from "#/components/shared/navigation-link";
import { I18nKey } from "#/i18n/declaration";
import { useBackendScopedPath } from "#/hooks/use-backend-scoped-path";
import {
  getInsiderConversationPath,
  isInsiderConversation,
} from "#/utils/insider-cat";
import { cn } from "#/utils/utils";

interface InsiderCatBadgeProps {
  tags?: Record<string, string> | null;
  parentConversationId?: string | null;
  /** A conversation ID turns the badge into a link back to its Cat page. */
  conversationId?: string;
  compact?: boolean;
}

export function InsiderCatBadge({
  tags,
  parentConversationId,
  conversationId,
  compact = false,
}: InsiderCatBadgeProps) {
  const { t } = useTranslation("openhands");
  const backendScopedPath = useBackendScopedPath();
  if (
    !isInsiderConversation({
      tags,
      parent_conversation_id: parentConversationId,
    })
  )
    return null;

  const badge = (
    <span
      data-testid="insider-cat-badge"
      className={cn(
        "inline-flex shrink-0 items-center gap-1.5 text-[var(--oh-accent)]",
        !compact &&
          "rounded-md border border-[var(--oh-accent)]/25 bg-[var(--oh-accent)]/10 px-1.5 py-0.5 text-[11px] font-medium leading-4",
      )}
      title={t(I18nKey.CONVERSATION$INSIDER_CAT)}
    >
      <Cat size={compact ? 17 : 14} aria-hidden="true" />
      <span className={compact ? "sr-only" : undefined}>
        {t(I18nKey.CONVERSATION$INSIDER_CAT)}
      </span>
    </span>
  );

  if (!conversationId) return badge;
  return (
    <NavigationLink
      to={backendScopedPath(getInsiderConversationPath(conversationId))}
      aria-label={t(I18nKey.CONVERSATION$OPEN_INSIDER_CAT)}
      title={t(I18nKey.CONVERSATION$OPEN_INSIDER_CAT)}
      className="shrink-0 rounded-md transition-opacity hover:opacity-80 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[var(--oh-accent)]"
    >
      {badge}
    </NavigationLink>
  );
}
