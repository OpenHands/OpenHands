import { beforeEach, describe, expect, it } from "vitest";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";
import type { SlashCommandItem } from "#/types/slash-command";

const resources = {
  skills: [{ name: "review", description: "Review code", source: "project" }],
  hooks: [{ hookType: "pre_tool_use", commands: ["lint"] }],
  mcps: [{ name: "github", transport: "stdio" }],
};

const entries = (scopeId = "conversation-1") =>
  useSlashCommandOutputStore.getState().entriesByScope[scopeId] ?? [];

const helpCommand = (command: string): SlashCommandItem => ({
  command,
  skill: {
    name: command.slice(1),
    type: "agentskills",
    source: null,
  },
});

describe("slash-command output store", () => {
  beforeEach(() => {
    useSlashCommandOutputStore.setState({
      entriesByScope: {},
      nextInvocationOrder: 0,
    });
  });

  it("begins /skills synchronously and returns the loading entry ID", () => {
    const id = useSlashCommandOutputStore
      .getState()
      .beginSkills("conversation-1", "event-1");

    expect(id).toEqual(expect.any(String));
    expect(entries()).toEqual([
      expect.objectContaining({
        id,
        kind: "skills",
        status: "loading",
        invocationOrder: 0,
        timelineBoundaryEventId: "event-1",
        showWhenPlacementUnresolved: true,
      }),
    ]);
  });

  it("completes the same entry without changing its identity or placement", () => {
    const store = useSlashCommandOutputStore.getState();
    const id = store.beginSkills("conversation-1", "event-1", 7);

    store.completeSkills("conversation-1", id, resources);

    expect(entries()).toHaveLength(1);
    expect(entries()[0]).toEqual(
      expect.objectContaining({
        id,
        status: "ready",
        invocationOrder: 7,
        timelineBoundaryEventId: "event-1",
        resources,
      }),
    );
  });

  it.each(["request", "timeout"] as const)(
    "persists a %s failure on the same entry",
    (errorKind) => {
      const store = useSlashCommandOutputStore.getState();
      const id = store.beginSkills("conversation-1", "event-1");

      store.failSkills("conversation-1", id, errorKind);

      expect(entries()).toEqual([
        expect.objectContaining({ id, status: "error", errorKind }),
      ]);
    },
  );

  it("ignores late completion after a terminal timeout", () => {
    const store = useSlashCommandOutputStore.getState();
    const id = store.beginSkills("conversation-1", "event-1");
    store.failSkills("conversation-1", id, "timeout");

    store.completeSkills("conversation-1", id, resources);

    expect(entries()).toEqual([
      expect.objectContaining({ id, status: "error", errorKind: "timeout" }),
    ]);
  });

  it("does not recreate an entry cleared before settlement", () => {
    const store = useSlashCommandOutputStore.getState();
    const id = store.beginSkills("conversation-1", "event-1");
    store.clear("conversation-1");

    store.completeSkills("conversation-1", id, resources);
    store.failSkills("conversation-1", id, "request");

    expect(entries()).toEqual([]);
  });

  it("keeps two invocations distinct and ordered when the second finishes first", () => {
    const store = useSlashCommandOutputStore.getState();
    const firstId = store.beginSkills("conversation-1", "event-1");
    const secondId = store.beginSkills("conversation-1", "event-1");

    store.completeSkills("conversation-1", secondId, resources);
    store.failSkills("conversation-1", firstId, "timeout");

    expect(entries().map((entry) => entry.id)).toEqual([firstId, secondId]);
    expect(entries().map((entry) => entry.invocationOrder)).toEqual([0, 1]);
    expect(entries()).toEqual([
      expect.objectContaining({ status: "error" }),
      expect.objectContaining({ status: "ready" }),
    ]);
  });

  it("deactivates only unresolved-placement fallbacks for the requested scope", () => {
    const store = useSlashCommandOutputStore.getState();
    store.beginSkills("conversation-1", "event-1");
    store.beginSkills("conversation-2", "event-2");

    store.deactivateSkillsPlacementFallback("conversation-1");

    expect(entries("conversation-1")[0]).toMatchObject({
      showWhenPlacementUnresolved: false,
    });
    expect(entries("conversation-2")[0]).toMatchObject({
      showWhenPlacementUnresolved: true,
    });
  });

  it("begins /help with fallback commands and completes the same entry", () => {
    const store = useSlashCommandOutputStore.getState();
    const fallback = [helpCommand("/help")];
    const enriched = [...fallback, helpCommand("/review")];
    const id = store.beginHelp("conversation-1", "event-1", fallback, 4);

    expect(entries()).toEqual([
      expect.objectContaining({
        id,
        kind: "help",
        status: "loading",
        commands: fallback,
        invocationOrder: 4,
        timelineBoundaryEventId: "event-1",
      }),
    ]);

    store.completeHelp("conversation-1", id, enriched);

    expect(entries()).toEqual([
      expect.objectContaining({
        id,
        status: "ready",
        commands: enriched,
        invocationOrder: 4,
        timelineBoundaryEventId: "event-1",
      }),
    ]);
  });

  it("keeps fallback help terminal after failure and ignores late completion", () => {
    const store = useSlashCommandOutputStore.getState();
    const fallback = [helpCommand("/help")];
    const id = store.beginHelp("conversation-1", "event-1", fallback);

    store.failHelp("conversation-1", id);
    store.completeHelp("conversation-1", id, [
      ...fallback,
      helpCommand("/review"),
    ]);

    expect(entries()).toEqual([
      expect.objectContaining({
        id,
        status: "ready",
        commands: fallback,
      }),
    ]);
  });

  it("keeps concurrent /help entries ordered through reverse completion", () => {
    const store = useSlashCommandOutputStore.getState();
    const firstId = store.beginHelp("conversation-1", "event-1", []);
    const secondId = store.beginHelp("conversation-1", "event-2", []);

    store.completeHelp("conversation-1", secondId, [helpCommand("/second")]);
    store.completeHelp("conversation-1", firstId, [helpCommand("/first")]);

    expect(entries().map((entry) => entry.id)).toEqual([firstId, secondId]);
    expect(entries().map((entry) => entry.invocationOrder)).toEqual([0, 1]);
    expect(entries().map((entry) => entry.status)).toEqual(["ready", "ready"]);
  });

  it("preserves /help and synchronous home /skills behavior", () => {
    const store = useSlashCommandOutputStore.getState();
    store.showHelp("home", null, []);
    store.showSkills("home", null, { skills: [], hooks: [], mcps: [] });

    expect(entries("home")).toEqual([
      expect.objectContaining({ kind: "help", status: "ready" }),
      expect.objectContaining({
        kind: "skills",
        status: "ready",
        showWhenPlacementUnresolved: false,
      }),
    ]);
  });
});
