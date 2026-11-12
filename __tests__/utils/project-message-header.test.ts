import { describe, expect, it } from "vitest";
import { parseProjectMessageHeader } from "#/utils/project-message-header";

const ID = "3cd1a23c000040008000000000000001";
const SENDER = `conversation:${ID}`;

describe("parseProjectMessageHeader", () => {
  it("reads an update header with a mention and keeps the body", () => {
    const text = `[Update from [Fix export](conversation:${ID}), a conversation in your project, not from the person. It finished its turn; its final message follows.]\nAll done.`;
    expect(parseProjectMessageHeader(text, SENDER)).toEqual({
      kind: "update_from",
      title: "Fix export",
      conversationId: ID,
      body: "All done.",
    });
  });

  it("reads the other header kinds", () => {
    const kinds = {
      "Sent by": "sent_by",
      "Reply in": "reply_in",
      "Message from the person in": "person_in",
    } as const;
    for (const [prefix, kind] of Object.entries(kinds)) {
      const parsed = parseProjectMessageHeader(
        `[${prefix} [T](conversation:${ID}), more.]\nbody`,
        SENDER,
      );
      expect(parsed?.kind).toBe(kind);
      expect(parsed?.body).toBe("body");
    }
  });

  it("names a conversation without a mention by its title", () => {
    const parsed = parseProjectMessageHeader(
      "[Sent by Coordinator, the coordinator of your project, not by the person.]\nGo",
      SENDER,
    );
    expect(parsed).toMatchObject({
      kind: "sent_by",
      title: "Coordinator",
      conversationId: null,
      body: "Go",
    });
  });

  it("leaves the user's own messages alone", () => {
    const text = `[Update from [X](conversation:${ID}), hi]\nbody`;
    expect(parseProjectMessageHeader(text, undefined)).toBeNull();
    expect(parseProjectMessageHeader(text, "user")).toBeNull();
    expect(parseProjectMessageHeader("plain text", SENDER)).toBeNull();
  });
});
