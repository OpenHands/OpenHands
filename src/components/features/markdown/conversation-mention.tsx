import React from "react";
import { ExtraProps } from "react-markdown";
import { NavigationLink } from "#/components/shared/navigation-link";
import { useIsExperimentalFeatureEnabled } from "#/hooks/query/use-experimental-features";
import { useProjectsEnabled } from "#/hooks/query/use-projects";

/** The Agent Server experimental feature that turns mentions on. */
export const CONVERSATION_MENTIONS_FEATURE_KEY = "conversation_mentions";

/**
 * A conversation mention, as Agent Server and agents write it:
 * `[Title](conversation:<id hex>)`. The id may also carry dashes.
 */
const CONVERSATION_MENTION =
  /^conversation:([0-9a-f]{8}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{4}-?[0-9a-f]{12})$/i;

/** The conversation id a link points at, or null for any other link. */
export function conversationMentionId(href: string | undefined | null) {
  const match = CONVERSATION_MENTION.exec(href ?? "");
  return match ? match[1].toLowerCase().replace(/-/g, "") : null;
}

const MENTION_TITLE_MAX = 80;

/**
 * The mention of a conversation, as Agent Server writes it. The title is cut
 * to one line without brackets, parentheses, quotes or backslashes so it
 * cannot end the link early (same rule as the SDK's `conversation_mention`).
 */
export function conversationMention(title: string | null, id: string) {
  const text =
    (title ?? "")
      .replace(/[\s"[\]()\\`]+/g, " ")
      .trim()
      .slice(0, MENTION_TITLE_MAX)
      .trim() || "untitled conversation";
  return `[${text}](conversation:${id.toLowerCase().replace(/-/g, "")})`;
}

/**
 * Keep conversation mentions as written; react-markdown drops links with
 * schemes it does not know.
 */
export function markdownUrlTransform(
  url: string,
  fallback: (url: string) => string,
) {
  return conversationMentionId(url) ? url : fallback(url);
}

export function ConversationMention({
  id,
  children,
}: {
  id: string;
  children: React.ReactNode;
}) {
  // While the feature is off a mention reads as its title, plain text.
  // Projects' header lines name conversations by mention, so Projects
  // turns them on too.
  const mentionsEnabled = useIsExperimentalFeatureEnabled(
    CONVERSATION_MENTIONS_FEATURE_KEY,
  );
  const projectsEnabled = useProjectsEnabled();
  if (!mentionsEnabled && !projectsEnabled) {
    return <span data-testid="conversation-mention-text">{children}</span>;
  }
  return (
    <NavigationLink
      to={`/conversations/${id}`}
      className="text-link underline"
      data-testid="conversation-mention"
      title={id}
    >
      {children}
    </NavigationLink>
  );
}

/**
 * Wraps a markdown `a` renderer: conversation mentions open the conversation
 * in place, every other link goes to `render` (or a plain anchor).
 */
export function withConversationMentions(render?: React.ElementType) {
  function MarkdownLink({
    node,
    ...props
  }: React.AnchorHTMLAttributes<HTMLAnchorElement> & ExtraProps) {
    const id = conversationMentionId(props.href);
    if (id) {
      return (
        <ConversationMention id={id}>{props.children}</ConversationMention>
      );
    }
    if (render) {
      return React.createElement(render, { node, ...props });
    }
    const { children, ...rest } = props;
    return <a {...rest}>{children}</a>;
  }
  return MarkdownLink;
}
