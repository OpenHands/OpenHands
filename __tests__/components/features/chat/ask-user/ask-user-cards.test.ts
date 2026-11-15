import { describe, expect, it } from "vitest";
import { buildAskUserCards } from "#/components/features/chat/ask-user/ask-user-cards";
import {
  answer,
  answerMessage,
  askAction,
  askShown,
  QUESTION,
  userMessage,
} from "./ask-user-fixtures";

describe("buildAskUserCards", () => {
  it("reads a shown question as waiting, under its observation", () => {
    const cards = buildAskUserCards([askAction(), askShown()]);

    expect(cards.byObservationId.get("obs-ask-1")).toEqual({
      askId: "ask-1",
      question: QUESTION,
      options: [
        { label: "Postgres", consequence: "Needs a managed database." },
        { label: "SQLite", consequence: "One file, no server." },
      ],
      recommended: 0,
      status: { kind: "pending" },
    });
  });

  it("records a tap and hides the message it sent", () => {
    const cards = buildAskUserCards([
      askAction(),
      askShown(),
      answer("ask-1", 3, { text: "use port 5433" }),
      answerMessage(),
    ]);

    expect(cards.byObservationId.get("obs-ask-1")?.status).toEqual({
      kind: "answered",
      optionIndex: 1,
      optionLabel: "SQLite",
      text: "use port 5433",
    });
    expect([...cards.answerMessageIds]).toEqual(["answer-msg"]);
  });

  it("closes a question the user answered in the chat, and keeps that message", () => {
    const cards = buildAskUserCards([
      askAction(),
      askShown(),
      userMessage("chat-1", "Neither yet, use Postgres with an env var.", 3),
    ]);

    expect(cards.byObservationId.get("obs-ask-1")?.status).toEqual({
      kind: "answered_in_chat",
    });
    expect(cards.answerMessageIds.size).toBe(0);
  });

  it("is not answered by a message relayed from another conversation", () => {
    const cards = buildAskUserCards([
      askAction(),
      askShown(),
      userMessage(
        "relay-1",
        "Update from the coordinator",
        3,
        "conversation:abc",
      ),
    ]);

    expect(cards.byObservationId.get("obs-ask-1")?.status.kind).toBe("pending");
  });

  it("keeps only the latest question waiting", () => {
    const cards = buildAskUserCards([
      askAction("ask-1", 1),
      askShown("ask-1", 2),
      askAction("ask-2", 3),
      askShown("ask-2", 4),
    ]);

    expect(cards.byObservationId.get("obs-ask-1")?.status.kind).toBe("closed");
    expect(cards.byObservationId.get("obs-ask-2")?.status.kind).toBe("pending");
  });

  it("makes no card for a refused question", () => {
    const cards = buildAskUserCards([askAction(), askShown("ask-1", 2, true)]);

    expect(cards.byObservationId.size).toBe(0);
  });
});
