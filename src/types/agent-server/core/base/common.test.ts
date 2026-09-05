import { describe, expect, expectTypeOf, it } from "vitest";
import {
  ExecutionStatus,
  ImageContent,
  SourceType,
  TextContent,
} from "#/types/agent-server/core/base/common";
import type {
  ImageContent as CanonicalImageContent,
  TextContent as CanonicalTextContent,
} from "@openhands/typescript-client";
import {
  isBaseEvent,
  isAgentServerEvent,
} from "#/types/agent-server/type-guards";

describe("ExecutionStatus (sourced from @openhands/typescript-client)", () => {
  it("keeps every historical status value stable", () => {
    expect(ExecutionStatus.IDLE).toBe("idle");
    expect(ExecutionStatus.RUNNING).toBe("running");
    expect(ExecutionStatus.PAUSED).toBe("paused");
    expect(ExecutionStatus.WAITING_FOR_CONFIRMATION).toBe(
      "waiting_for_confirmation",
    );
    expect(ExecutionStatus.FINISHED).toBe("finished");
    expect(ExecutionStatus.ERROR).toBe("error");
    expect(ExecutionStatus.STUCK).toBe("stuck");
  });

  it("exposes server statuses the old local enum was missing", () => {
    // The canonical client contract adds "deleting" to the historical set.
    // Canvas previously redeclared ExecutionStatus without it, drifting from
    // the wire model (#16952).
    expect(ExecutionStatus.DELETING).toBe("deleting");
  });
});

describe("SourceType (sourced from @openhands/typescript-client)", () => {
  it("is equivalent to the canonical EventSource", () => {
    const sources: readonly SourceType[] = [
      "agent",
      "user",
      "environment",
      "system",
      "hook",
    ];
    expect(sources).toContain("system");
  });
});

describe("runtime guards and the widened source set", () => {
  it("isBaseEvent accepts a system-sourced event", () => {
    const systemEvent = {
      id: "01JW7T4XG8Q0VPRQM6YK0N3ZB2",
      timestamp: "2026-08-27T00:00:00.000Z",
      source: "system",
      kind: "SystemPromptEvent",
    };
    expect(isBaseEvent(systemEvent)).toBe(true);
    expect(isAgentServerEvent(systemEvent)).toBe(true);
  });

  it("isBaseEvent still rejects non-events and unknown sources", () => {
    const notAnEvent = { id: "id", timestamp: "ts", source: "unknown" };
    expect(isBaseEvent(notAnEvent)).toBe(false);
    expect(isBaseEvent(null)).toBe(false);
    expect(isBaseEvent("string")).toBe(false);
  });
});

describe("TextContent / ImageContent (sourced from @openhands/typescript-client)", () => {
  it("refines the canonical client content types instead of redeclaring them", () => {
    // The local types must be assignable both ways against the canonical
    // client contracts so Canvas stays a pure consumer of the wire model.
    const text: TextContent = { type: "text", text: "hello" };
    const image: ImageContent = { type: "image", image_urls: ["u1", "u2"] };
    expectTypeOf<TextContent>().toMatchTypeOf<CanonicalTextContent>();
    expectTypeOf<ImageContent>().toMatchTypeOf<CanonicalImageContent>();
    expect(text.type).toBe("text");
    expect(image.image_urls).toHaveLength(2);
  });

  it("keeps the wire's optional cache_prompt refinement", () => {
    // The agent-server schema emits `cache_prompt` on content blocks, but the
    // client's simplified content types omit it. Canvas preserves the narrow
    // refinement so wire payloads with the flag still type-check (#16952).
    const cachedText: TextContent = {
      type: "text",
      text: "cached",
      cache_prompt: true,
    };
    const uncachedImage: ImageContent = {
      type: "image",
      image_urls: ["u"],
      cache_prompt: false,
    };
    expect(cachedText.cache_prompt).toBe(true);
    expect(uncachedImage.cache_prompt).toBe(false);
  });

  it("preserves the discriminated structural base from the client", () => {
    const blocks: (TextContent | ImageContent)[] = [
      { type: "text", text: "a" },
      { type: "image", image_urls: ["b"] },
    ];
    const texts = blocks.filter((b) => b.type === "text");
    const images = blocks.filter((b) => b.type === "image");
    // Narrowing on `type` works because the canonical base is a union.
    expect(texts[0].text).toBe("a");
    expect(images[0].image_urls).toEqual(["b"]);
  });
});
