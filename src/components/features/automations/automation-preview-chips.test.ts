import { describe, expect, it } from "vitest";
import {
  previewChipItems,
  shouldChipPreviewField,
} from "./automation-preview-chips";

describe("automation preview chips", () => {
  it("chips repositories, plugins, feeds, and topics only", () => {
    expect(shouldChipPreviewField("repository", "repo-picker")).toBe(true);
    expect(shouldChipPreviewField("plugins", "plugins")).toBe(true);
    expect(shouldChipPreviewField("feeds", "textarea")).toBe(true);
    expect(shouldChipPreviewField("topics", "textarea")).toBe(true);
    expect(shouldChipPreviewField("notes", "textarea")).toBe(false);
    expect(shouldChipPreviewField("prompt", "prompt")).toBe(false);
    expect(shouldChipPreviewField("webhook", "text")).toBe(false);
    expect(shouldChipPreviewField("name", "name")).toBe(false);
  });

  it("splits feeds on lines and topics on lines or commas", () => {
    expect(
      previewChipItems("feeds", "textarea", [
        "https://a.example/rss\nhttps://b.example/rss",
      ]),
    ).toEqual(["https://a.example/rss", "https://b.example/rss"]);
    expect(
      previewChipItems("topics", "textarea", ["ai, open source\nsecurity"]),
    ).toEqual(["ai", "open source", "security"]);
  });
});
