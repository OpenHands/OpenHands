import { describe, expect, it } from "vitest";
import {
  isPreviewIdentityField,
  sortPreviewFields,
  type PreviewFieldKind,
} from "./automation-preview-order";

const kinds: Record<string, PreviewFieldKind> = {
  title: "text",
  schedule: "cron",
  timezone: "timezone",
  repository: "repo-picker",
  webhook: "text",
  environment: "select",
  notes: "textarea",
  feeds: "textarea",
  prompt: "prompt",
  topics: "textarea",
  summary: "textarea",
  plugins: "plugins",
};

describe("automation preview field order", () => {
  it("bubbles identity, when, and where; sinks long body and plugins", () => {
    const names = sortPreviewFields(
      [
        "plugins",
        "notes",
        "feeds",
        "repository",
        "timezone",
        "schedule",
        "title",
        "webhook",
        "environment",
        "prompt",
        "topics",
        "summary",
      ],
      (name) => name,
      (name) => kinds[name],
    );

    expect(names).toEqual([
      "title",
      "schedule",
      "timezone",
      "repository",
      "webhook",
      "environment",
      "notes",
      "feeds",
      "prompt",
      "topics",
      "summary",
      "plugins",
    ]);
  });

  it("keeps declaration order when two fields share a rank", () => {
    const names = sortPreviewFields(
      ["repoB", "repoA"],
      (name) => name,
      () => "repo-picker",
    );

    expect(names).toEqual(["repoB", "repoA"]);
  });

  it("treats name and title as identity fields", () => {
    expect(isPreviewIdentityField("name")).toBe(true);
    expect(isPreviewIdentityField("title")).toBe(true);
    expect(isPreviewIdentityField("widgetName")).toBe(false);
  });
});
