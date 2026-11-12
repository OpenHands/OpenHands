/**
 * Messages one conversation sends another in a project open with a bracketed
 * header line written by the Agent Server, such as
 * `[Update from [Title](conversation:<hex>), a conversation in your project, …]`.
 * The header is for the agent; the user sees a compact line instead.
 */
export type ProjectHeaderKind =
  "sent_by" | "update_from" | "reply_in" | "person_in";

const KINDS: Record<string, ProjectHeaderKind> = {
  "Sent by": "sent_by",
  "Update from": "update_from",
  "Reply in": "reply_in",
  "Message from the person in": "person_in",
};

const HEADER =
  /^\[(Sent by|Update from|Reply in|Message from the person in) (?:\[([^\]\n]*)\]\(conversation:([0-9a-f-]{32,36})\)|([^,\n]*))[^\n]*\]\s*\n?/i;

export interface ProjectMessageHeader {
  kind: ProjectHeaderKind;
  /** The other conversation's title, as the header names it. */
  title: string;
  /** Its id in plain hex, when the header carries a mention. */
  conversationId: string | null;
  /** The message without the header line. */
  body: string;
}

/**
 * The header of a message another conversation sent (`sender` is
 * `conversation:<hex>`), or null for anything else, including the user's
 * own messages that merely start with a bracket.
 */
export function parseProjectMessageHeader(
  text: string,
  sender: unknown,
): ProjectMessageHeader | null {
  if (typeof sender !== "string" || !sender.startsWith("conversation:")) {
    return null;
  }
  const match = HEADER.exec(text);
  if (!match) return null;
  const id = match[3] ? match[3].replace(/-/g, "").toLowerCase() : null;
  return {
    kind: KINDS[match[1]] ?? "sent_by",
    title: (match[2] ?? match[4] ?? "").trim(),
    conversationId: id?.length === 32 ? id : null,
    body: text.slice(match[0].length),
  };
}
