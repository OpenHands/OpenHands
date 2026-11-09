import React from "react";
import { ExtraProps } from "react-markdown";
import { NavigationLink } from "#/components/shared/navigation-link";

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
