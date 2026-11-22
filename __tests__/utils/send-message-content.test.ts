import { describe, expect, it } from "vitest";
import type { TFunction } from "i18next";
import { buildUserMessageContent } from "#/utils/send-message-content";

const t = ((key: string) => `translated:${key}`) as unknown as TFunction;

describe("buildUserMessageContent", () => {
  it("builds a text-only content array", () => {
    expect(buildUserMessageContent({ text: "Explain this code", t })).toEqual([
      { type: "text", text: "Explain this code" },
    ]);
  });

  it("appends one image block after the text when images are attached", () => {
    expect(
      buildUserMessageContent({
        text: "What is in this picture?",
        imageUrls: ["data:image/png;base64,first"],
        t,
      }),
    ).toEqual([
      { type: "text", text: "What is in this picture?" },
      { type: "image", image_urls: ["data:image/png;base64,first"] },
    ]);
  });

  it("augments the prompt with uploaded file references", () => {
    expect(
      buildUserMessageContent({
        text: "Please inspect these files",
        fileUrls: ["/workspace/notes.txt", "/workspace/large.png"],
        t,
      }),
    ).toEqual([
      {
        type: "text",
        text: "Please inspect these files\n\ntranslated:CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE: /workspace/notes.txt\n\n/workspace/large.png",
      },
    ]);
  });

  it("keeps the wire format of the REST path for file-only messages", () => {
    const content = buildUserMessageContent({
      text: "",
      fileUrls: ["/workspace/notes.txt"],
      t,
    });

    expect(content).toEqual([
      {
        type: "text",
        text: "\n\ntranslated:CHAT_INTERFACE$AUGMENTED_PROMPT_FILES_TITLE: /workspace/notes.txt",
      },
    ]);
  });

  it("keeps the empty text part for image-only messages (wire parity)", () => {
    const content = buildUserMessageContent({
      text: "",
      imageUrls: ["data:image/png;base64,first"],
      t,
    });

    expect(content).toEqual([
      { type: "text", text: "" },
      { type: "image", image_urls: ["data:image/png;base64,first"] },
    ]);
  });

  it("rejects a message with no text and no attachments", () => {
    expect(() => buildUserMessageContent({ text: "", t })).toThrow();
    expect(() =>
      buildUserMessageContent({
        text: "",
        imageUrls: [],
        fileUrls: [],
        t,
      }),
    ).toThrow();
  });
});
