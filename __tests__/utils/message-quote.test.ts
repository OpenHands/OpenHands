import { describe, expect, it } from "vitest";

import type { MessageEvent } from "#/types/agent-server/core";
import {
  REPLY_CLOSE_TAG,
  REPLY_OPEN_TAG,
  composeReplyText,
  quoteBodyForEvent,
  splitReplyFromText,
  type ReplyQuote,
} from "#/utils/message-quote";

const sampleQuote: ReplyQuote = {
  sourceEventId: "01JABC123",
  source: "agent",
  body: "Fix the login bug",
  truncated: false,
};

const makeEvent = (attrs: {
  id: string;
  source: "user" | "agent";
  text?: string;
}): MessageEvent => ({
  id: attrs.id,
  timestamp: "2026-10-28T00:00:00Z",
  source: attrs.source,
  llm_message: {
    role: attrs.source === "user" ? "user" : "assistant",
    content: [{ type: "text", text: attrs.text ?? "" }],
  },
  activated_skills: [],
  extended_content: [],
});

describe("message-quote constants", () => {
  it("exposes the spec wire-format tokens as named constants", () => {
    expect(REPLY_OPEN_TAG).toBe("<reply-to");
    expect(REPLY_CLOSE_TAG).toBe("</reply-to>");
  });
});

describe("composeReplyText", () => {
  it("composes start-anchored envelope plus follow-up", () => {
    const text = composeReplyText(sampleQuote, "please look again");

    expect(text).toBe(
      `<reply-to id="01JABC123" source="agent" truncated="false">Fix the login bug</reply-to>please look again`,
    );
  });

  it("composes just the envelope when the follow-up is blank", () => {
    const text = composeReplyText(sampleQuote, "   ");
    expect(text).toBe(
      `<reply-to id="01JABC123" source="agent" truncated="false">Fix the login bug</reply-to>`,
    );
  });

  it("marks truncated quotes in the envelope flag", () => {
    const text = composeReplyText({ ...sampleQuote, truncated: true }, "go on");
    expect(text).toContain('truncated="true"');
  });

  it("returns the follow-up unchanged when there is no quote", () => {
    expect(composeReplyText(null, "just text")).toBe("just text");
  });

  it("is idempotent — a follow-up already starting with the envelope is untouched", () => {
    const once = composeReplyText(sampleQuote, "please look again");
    expect(composeReplyText(sampleQuote, once)).toBe(once);
  });
});

describe("splitReplyFromText", () => {
  it("round-trips a composed envelope back into quote and body", () => {
    const text = composeReplyText(sampleQuote, "please look again");

    expect(splitReplyFromText(text)).toEqual({
      quote: sampleQuote,
      body: "please look again",
    });
  });

  it("parses an envelope-only message into an empty body", () => {
    const text = composeReplyText(sampleQuote, "");

    expect(splitReplyFromText(text)).toEqual({
      quote: sampleQuote,
      body: "",
    });
  });

  it("is anchored at position 0 — an envelope later in the text stays verbatim", () => {
    const text = `see this: <reply-to id="x" source="user" truncated="false">hi</reply-to> done`;

    expect(splitReplyFromText(text)).toEqual({ quote: null, body: text });
  });

  it("returns plain text unchanged when no envelope is present", () => {
    const text = "ordinary message\nwith lines";

    expect(splitReplyFromText(text)).toEqual({ quote: null, body: text });
  });

  it("degrades a malformed envelope to plain text instead of crashing", () => {
    const malformed = `<reply-to id="oops" source="robot" truncated="maybe">body never closed`;

    expect(splitReplyFromText(malformed)).toEqual({
      quote: null,
      body: malformed,
    });
  });

  it("strips an inner envelope when splitting a reply-of-a-reply", () => {
    const inner = composeReplyText(
      {
        sourceEventId: "old",
        source: "user",
        body: "earlier",
        truncated: false,
      },
      "first reply",
    );
    const outer = composeReplyText(
      { sourceEventId: "new", source: "agent", body: inner, truncated: false },
      "second reply",
    );

    const split = splitReplyFromText(outer);
    expect(split.body).toBe("second reply");
    expect(split.quote?.body).toBe("first reply");
  });
});
describe("quoteBodyForEvent", () => {
  it("derives a quote from a user event's displayed text", () => {
    const event = makeEvent({
      id: "evt-1",
      source: "user",
      text: "hello there",
    });

    expect(quoteBodyForEvent(event, "hello there")).toEqual({
      sourceEventId: "evt-1",
      source: "user",
      body: "hello there",
      truncated: false,
    });
  });

  it("truncates bodies beyond the 2000-char cap and flags it", () => {
    const long = "x".repeat(2500);
    const event = makeEvent({ id: "evt-2", source: "agent", text: long });

    const quote = quoteBodyForEvent(event, long);
    expect(quote?.truncated).toBe(true);
    expect(quote?.body.length).toBe(2000);
  });

  it("keeps truncated=false at exactly the cap", () => {
    const exact = "y".repeat(2000);
    const event = makeEvent({ id: "evt-3", source: "agent", text: exact });

    expect(quoteBodyForEvent(event, exact)?.truncated).toBe(false);
  });

  it("strips an inner envelope from the quoted body", () => {
    const innerEnvelope = `<reply-to id="old" source="user" truncated="false">earlier</reply-to>real text`;
    const event = makeEvent({
      id: "evt-4",
      source: "user",
      text: innerEnvelope,
    });

    expect(quoteBodyForEvent(event, innerEnvelope)?.body).toBe("real text");
  });

  it("returns null for image-only or empty sources", () => {
    const empty = makeEvent({ id: "evt-5", source: "user", text: "" });
    expect(quoteBodyForEvent(empty, "")).toBeNull();

    const whitespace = makeEvent({ id: "evt-6", source: "user", text: "   " });
    expect(quoteBodyForEvent(whitespace, "   ")).toBeNull();
  });
});
