import { describe, expect, it } from "vitest";
import { resolveSlashCommandOutputPlacements } from "#/components/conversation-events/chat/slash-command-output-placement";
import type { SlashCommandOutput } from "#/stores/slash-command-output-store";
import type { OpenHandsEvent } from "#/types/agent-server/core";
import { createUserMessageEvent } from "test-utils";

const helpOutput = (
  id: string,
  timelineBoundaryEventId: string | null,
  invocationOrder = 0,
): SlashCommandOutput => ({
  id,
  kind: "help",
  invocationOrder,
  timelineBoundaryEventId,
  commands: [],
});

const skillsOutput = (
  id: string,
  timelineBoundaryEventId: string,
  status: "loading" | "ready" = "loading",
  showWhenPlacementUnresolved = true,
): SlashCommandOutput =>
  status === "loading"
    ? {
        id,
        kind: "skills",
        status,
        invocationOrder: 0,
        timelineBoundaryEventId,
        showWhenPlacementUnresolved,
      }
    : {
        id,
        kind: "skills",
        status,
        invocationOrder: 0,
        timelineBoundaryEventId,
        showWhenPlacementUnresolved,
        resources: { skills: [], hooks: [], mcps: [] },
      };

describe("resolveSlashCommandOutputPlacements", () => {
  it("places output before the first rendered event after a replaced boundary", () => {
    const before = createUserMessageEvent("before");
    const replacedAction = {
      ...createUserMessageEvent("action"),
      source: "agent",
    } as OpenHandsEvent;
    const replacementObservation = {
      ...createUserMessageEvent("observation"),
      source: "environment",
    } as OpenHandsEvent;

    const placements = resolveSlashCommandOutputPlacements(
      [helpOutput("help", "action")],
      [before, replacedAction, replacementObservation],
      [before, replacementObservation],
    );

    expect(placements.entriesBeforeEvent.get("observation")).toEqual([
      expect.objectContaining({ id: "help" }),
    ]);
    expect(placements.breakBeforeEventIds).toEqual(new Set(["observation"]));
    expect(placements.tailEntries).toEqual([]);
  });

  it("preserves invocation order for equal boundaries at the known tail", () => {
    const boundary = createUserMessageEvent("boundary");
    const first = helpOutput("first", "boundary", 0);
    const second = helpOutput("second", "boundary", 1);

    const placements = resolveSlashCommandOutputPlacements(
      [first, second],
      [boundary],
      [boundary],
    );

    expect(placements.tailEntries.map((entry) => entry.id)).toEqual([
      "first",
      "second",
    ]);
  });

  it("does not place output whose non-null boundary is not loaded", () => {
    const recent = createUserMessageEvent("recent");
    const placements = resolveSlashCommandOutputPlacements(
      [helpOutput("missing", "not-loaded")],
      [recent],
      [recent],
    );

    expect(placements.entriesBeforeEvent.size).toBe(0);
    expect(placements.tailEntries).toEqual([]);
    expect(placements.breakBeforeEventIds.size).toBe(0);
  });

  it("keeps a newly submitted unresolved /skills invocation visible", () => {
    const recent = createUserMessageEvent("recent");

    const placements = resolveSlashCommandOutputPlacements(
      [skillsOutput("skills", "temporarily-missing")],
      [recent],
      [recent],
    );

    expect(placements.unresolvedActiveEntries).toEqual([
      expect.objectContaining({ id: "skills", status: "loading" }),
    ]);
  });

  it("keeps a ready result visible until the active-view fallback is deactivated", () => {
    const recent = createUserMessageEvent("recent");
    const active = resolveSlashCommandOutputPlacements(
      [skillsOutput("skills", "temporarily-missing", "ready")],
      [recent],
      [recent],
    );
    const historical = resolveSlashCommandOutputPlacements(
      [skillsOutput("skills", "temporarily-missing", "ready", false)],
      [recent],
      [recent],
    );

    expect(active.unresolvedActiveEntries).toHaveLength(1);
    expect(historical.unresolvedActiveEntries).toEqual([]);
    expect(historical.tailEntries).toEqual([]);
  });

  it("reconciles an active fallback to its anchored position when pagination restores the boundary", () => {
    const boundary = createUserMessageEvent("boundary");
    const recent = createUserMessageEvent("recent");

    const placements = resolveSlashCommandOutputPlacements(
      [skillsOutput("skills", "boundary", "ready")],
      [boundary, recent],
      [boundary, recent],
    );

    expect(placements.unresolvedActiveEntries).toEqual([]);
    expect(placements.entriesBeforeEvent.get("recent")).toEqual([
      expect.objectContaining({ id: "skills" }),
    ]);
  });

  it("leaves null-boundary output to the empty-conversation/home renderer", () => {
    const placements = resolveSlashCommandOutputPlacements(
      [helpOutput("home", null)],
      [],
      [],
    );

    expect(placements.entriesBeforeEvent.size).toBe(0);
    expect(placements.tailEntries).toEqual([]);
  });
});
