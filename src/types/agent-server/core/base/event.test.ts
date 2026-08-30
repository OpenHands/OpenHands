import { describe, expect, expectTypeOf, it } from "vitest";
import type { BaseEvent } from "./event";
import type {
  BaseEvent as CanonicalBaseEvent,
  EventSource,
} from "@openhands/typescript-client";

describe("Canvas base event envelope (#16952 stage 2)", () => {
  it("sources the canonical typescript-client envelope instead of redeclaring it", () => {
    // Canvas pins the wire fields and inherits everything else from the client
    // contract, so every Canvas event satisfies the canonical envelope when the
    // client-only `kind` discriminator is ignored.
    expectTypeOf<BaseEvent>().toMatchTypeOf<Omit<CanonicalBaseEvent, "kind">>();
  });

  it("inherits canonical optional envelope fields (e.g. parent_id)", () => {
    // `parent_id` is not declared locally; it must come from the client
    // contract via the Omit+intersection above.
    const event: BaseEvent = {
      id: "evt-parent-links",
      timestamp: "2024-06-01T00:00:00Z",
      source: "user",
      parent_id: "evt-parent",
    };
    expect(event.parent_id).toBe("evt-parent");
  });

  it("accepts the full canonical event source set, including `system`", () => {
    const sources: EventSource[] = [
      "agent",
      "user",
      "environment",
      "system",
      "hook",
    ];
    for (const source of sources) {
      const event: BaseEvent = {
        id: `evt-${source}`,
        timestamp: "2024-06-01T00:00:00Z",
        source,
      };
      expect(event.source).toBe(source);
    }
  });

  it("keeps the wire fields required on the Canvas side", () => {
    // The canonical contract makes id/timestamp/source optional; Canvas pins
    // all three as required so consumers never handle a partial envelope.
    const event: BaseEvent = {
      id: "evt-required",
      timestamp: "2024-06-01T00:00:00Z",
      source: "agent",
    };
    expect(event).toEqual({
      id: "evt-required",
      timestamp: "2024-06-01T00:00:00Z",
      source: "agent",
    });
  });
});
