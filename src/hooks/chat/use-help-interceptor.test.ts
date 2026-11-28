import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useHelpInterceptor } from "./use-help-interceptor";
import { useSlashCommandInfoStore } from "#/stores/slash-command-info-store";

const mockUseConversationSkills = vi.fn();

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: (...args: unknown[]) =>
    mockUseConversationSkills(...args),
}));

const CONV = "conv-1";

const MOCK_SKILLS = [
  { name: "init", type: "agentskills", source: null, triggers: ["/init"] },
  { name: "releasenotes", type: "agentskills", source: null, triggers: [] },
  // Duplicates a built-in command: must not appear twice in /help.
  { name: "new", type: "agentskills", source: null, triggers: ["/new"] },
];

const setup = (conversationId: string | null) => {
  const onSubmit = vi.fn();
  const { result } = renderHook(() =>
    useHelpInterceptor(conversationId, onSubmit),
  );
  return { intercept: result.current, onSubmit };
};

const entries = () =>
  useSlashCommandInfoStore.getState().entriesByConversation[CONV] ?? [];

describe("useHelpInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSlashCommandInfoStore.setState({ entriesByConversation: {} });
    mockUseConversationSkills.mockReturnValue({ data: MOCK_SKILLS });
  });

  it("passes a non-help message straight through to onSubmit", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("hello there");
    expect(onSubmit).toHaveBeenCalledWith("hello there");
    expect(entries()).toHaveLength(0);
  });

  it("renders every built-in command with its description", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/help");
    expect(onSubmit).not.toHaveBeenCalled();
    expect(entries()).toHaveLength(1);
    const body = entries()[0].body;
    for (const command of [
      "/new",
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
      "/help",
      "/condense",
      "/skills",
      "/feedback",
    ]) {
      expect(body).toContain(`\`${command}\``);
    }
    expect(body).toContain("Display all available commands");
    // /new appears once: the skill-derived duplicate is filtered out.
    expect(body.match(/`\/new`/g)).toHaveLength(1);
  });

  it("lists skill-derived commands and the autocomplete note", () => {
    const { intercept } = setup(CONV);
    intercept("/help");
    const body = entries()[0].body;
    expect(body).toContain("`/init`");
    expect(body).toContain("`/releasenotes`");
    expect(body).toContain("Type `/` to show the autocomplete menu.");
  });

  it("notes when no skill commands are loaded", () => {
    mockUseConversationSkills.mockReturnValue({ data: [] });
    const { intercept } = setup(CONV);
    intercept("/help");
    expect(entries()[0].body).toContain("No skill commands loaded");
  });

  it("passes through (no entry) when there is no conversation", () => {
    const { intercept, onSubmit } = setup(null);
    intercept("/help");
    expect(onSubmit).toHaveBeenCalledWith("/help");
    expect(entries()).toHaveLength(0);
  });
});
