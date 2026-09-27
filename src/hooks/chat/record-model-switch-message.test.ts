import { describe, it, expect, beforeEach } from "vitest";
import { useModelStore } from "#/stores/model-store";
import {
  getStoredConversationMetadata,
  setStoredConversationMetadata,
} from "#/api/conversation-metadata-store";
import { OpenHandsEvent } from "#/types/agent-server/core";
import { seedModelSwitchesFromHistory } from "./record-model-switch-message";

const userMessage = (id: string): OpenHandsEvent =>
  ({
    id,
    timestamp: "2024-01-01T00:00:00Z",
    source: "user",
    llm_message: { role: "user", content: [{ type: "text", text: "hi" }] },
  }) as unknown as OpenHandsEvent;

const switchObservation = (
  id: string,
  profileName: string,
  isError = false,
  timestamp = "2024-01-01T00:00:00Z",
): OpenHandsEvent =>
  ({
    id,
    timestamp,
    source: "environment",
    action_id: `action-${id}`,
    observation: {
      kind: "SwitchLLMObservation",
      content: [],
      is_error: isError,
      profile_name: profileName,
      reason: null,
      active_model: null,
    },
  }) as unknown as OpenHandsEvent;

// An agent action event. `ThinkAction` is renderable (shown as a thinking
// block); `PlanningFileEditorAction` is hidden by `shouldRenderEvent`.
const agentAction = (id: string, kind: string): OpenHandsEvent =>
  ({
    id,
    timestamp: "2024-01-01T00:00:00Z",
    source: "agent",
    action: { kind, thought: "t" },
    tool_name: "tool",
    tool_call_id: `call-${id}`,
  }) as unknown as OpenHandsEvent;

const entriesFor = (conversationId: string) =>
  useModelStore.getState().entriesByConversation[conversationId] ?? [];

const activeProfileFor = (conversationId: string) =>
  useModelStore.getState().activeProfileByConversation[conversationId];

const stampedProfileFor = (conversationId: string) =>
  getStoredConversationMetadata(conversationId)?.active_profile;

describe("seedModelSwitchesFromHistory", () => {
  beforeEach(() => {
    useModelStore.getState().clearAll();
    window.localStorage.clear();
  });

  it("seeds a successful switch anchored to the prior renderable event", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].switchedTo).toBe("fast");
    expect(entries[0].anchorEventId).toBe("u1");
    expect(entries[0].id).toBe("history-switch:o1");
  });

  it("is idempotent across re-seeds (e.g. reloads)", () => {
    const events = [userMessage("u1"), switchObservation("o1", "fast")];
    seedModelSwitchesFromHistory("c1", events);
    seedModelSwitchesFromHistory("c1", events);

    expect(entriesFor("c1")).toHaveLength(1);
  });

  it("ignores failed switches (they still render as error cards)", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("e1", "fast", true),
    ]);

    expect(entriesFor("c1")).toHaveLength(0);
  });

  it("never anchors to a non-rendered event (must land on a rendered id)", () => {
    // PlanningFileEditorAction is hidden by shouldRenderEvent, so the renderer
    // never mounts it; anchoring there would orphan the message. The anchor
    // must fall back to the prior rendered event (the user message).
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      agentAction("p1", "PlanningFileEditorAction"),
      switchObservation("o1", "fast"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].anchorEventId).toBe("u1");
  });

  it("anchors to a renderable ThinkAction that precedes the switch", () => {
    // In uiEvents the ThinkObservation is dropped and the ThinkAction is kept
    // (and rendered as a thinking block), so it is a valid anchor.
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      agentAction("t1", "ThinkAction"),
      switchObservation("o1", "architect"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].anchorEventId).toBe("t1");
  });

  it("anchors to null when no renderable event precedes the switch", () => {
    seedModelSwitchesFromHistory("c1", [switchObservation("o1", "architect")]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(1);
    expect(entries[0].anchorEventId).toBeNull();
  });

  it("preserves order and anchors for multiple switches", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
      userMessage("u2"),
      switchObservation("o2", "architect"),
    ]);

    const entries = entriesFor("c1");
    expect(entries).toHaveLength(2);
    expect(entries[0]).toMatchObject({
      switchedTo: "fast",
      anchorEventId: "u1",
    });
    expect(entries[1]).toMatchObject({
      switchedTo: "architect",
      anchorEventId: "u2",
    });
  });

  it("re-stamps the active profile from a switch in the loaded history", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(activeProfileFor("c1")).toBe("fast");
    expect(stampedProfileFor("c1")).toBe("fast");
  });

  it("stamps the latest switch when history holds several", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
      userMessage("u2"),
      switchObservation("o2", "architect"),
    ]);

    expect(activeProfileFor("c1")).toBe("architect");
    expect(stampedProfileFor("c1")).toBe("architect");
  });

  it("leaves the stamp untouched when history has no switch", () => {
    setStoredConversationMetadata("c1", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: null,
      active_profile: "original",
      plugins: null,
    });
    useModelStore.getState().setActiveProfile("c1", "original");

    seedModelSwitchesFromHistory("c1", [userMessage("u1")]);

    expect(activeProfileFor("c1")).toBe("original");
    expect(stampedProfileFor("c1")).toBe("original");
  });

  it("does not stamp from a failed switch", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("e1", "fast", true),
    ]);

    expect(activeProfileFor("c1")).toBeUndefined();
    expect(stampedProfileFor("c1")).toBeUndefined();
  });

  it("keys the stamp per conversation", () => {
    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(activeProfileFor("c2")).toBeUndefined();
    expect(stampedProfileFor("c2")).toBeUndefined();
  });

  it("preserves the other stored metadata fields when re-stamping", () => {
    setStoredConversationMetadata("c1", {
      selected_repository: "org/repo",
      selected_branch: "main",
      git_provider: "github",
      selected_workspace: "/ws",
      active_profile: "stale",
      plugins: [{ source: "s", ref: "r", repo_path: null }],
    });

    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(getStoredConversationMetadata("c1")).toMatchObject({
      selected_repository: "org/repo",
      selected_branch: "main",
      git_provider: "github",
      selected_workspace: "/ws",
      active_profile: "fast",
      plugins: [{ source: "s", ref: "r", repo_path: null }],
    });
  });

  it("preserves workspace_mode across a stamp (#15520)", () => {
    setStoredConversationMetadata("c1", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      selected_workspace: "/ws",
      workspace_mode: "local_repo",
    });

    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(getStoredConversationMetadata("c1")?.workspace_mode).toBe(
      "local_repo",
    );
  });

  it("leaves a newer manual stamp alone (manual switch after a tool switch survives reload)", () => {
    // t1: agent tool-switches to "fast" (observation in history). t2: user
    // manually switches back to "architect" via /model — a REST call that
    // leaves no observation. The reload seed must not roll the stamp back.
    setStoredConversationMetadata("c1", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      active_profile: "architect",
      stamped_at: "2024-06-01T00:00:00Z",
    });

    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(getStoredConversationMetadata("c1")?.active_profile).toBe(
      "architect",
    );
    expect(getStoredConversationMetadata("c1")?.stamped_at).toBe(
      "2024-06-01T00:00:00Z",
    );
    // The in-memory stamp takes priority over the persisted one in the pill,
    // so it must not be set to the stale profile either.
    expect(activeProfileFor("c1")).toBeUndefined();
  });

  it("repairs a legacy stamp written before stamped_at existed", () => {
    setStoredConversationMetadata("c1", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      active_profile: "architect",
    });

    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(stampedProfileFor("c1")).toBe("fast");
    expect(getStoredConversationMetadata("c1")?.stamped_at).toBe(
      "2024-01-01T00:00:00Z",
    );
  });

  it("re-stamps when the latest observation is newer than the stored stamp", () => {
    setStoredConversationMetadata("c1", {
      selected_repository: null,
      selected_branch: null,
      git_provider: null,
      active_profile: "architect",
      stamped_at: "2023-01-01T00:00:00Z",
    });

    seedModelSwitchesFromHistory("c1", [
      userMessage("u1"),
      switchObservation("o1", "fast"),
    ]);

    expect(stampedProfileFor("c1")).toBe("fast");
    expect(getStoredConversationMetadata("c1")?.stamped_at).toBe(
      "2024-01-01T00:00:00Z",
    );
  });

  it("stamps with the observation timestamp so re-seeding the same history is a no-op", () => {
    const events = [userMessage("u1"), switchObservation("o1", "fast")];
    seedModelSwitchesFromHistory("c1", events);
    seedModelSwitchesFromHistory("c1", events);

    expect(getStoredConversationMetadata("c1")?.stamped_at).toBe(
      "2024-01-01T00:00:00Z",
    );
  });
});
