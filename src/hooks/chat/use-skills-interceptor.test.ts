import { renderHook } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { useSkillsInterceptor } from "./use-skills-interceptor";
import { useSlashCommandInfoStore } from "#/stores/slash-command-info-store";

const mockUseConversationSkills = vi.fn();

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: (...args: unknown[]) =>
    mockUseConversationSkills(...args),
}));

const CONV = "conv-1";

const MOCK_SKILLS = [
  {
    name: "init",
    type: "agentskills",
    source: null,
    description: "Initialise a repository",
    triggers: ["/init"],
  },
  { name: "releasenotes", type: "repo", source: null, triggers: [] },
];

const setup = (conversationId: string | null) => {
  const onSubmit = vi.fn();
  const { result } = renderHook(() =>
    useSkillsInterceptor(conversationId, onSubmit),
  );
  return { intercept: result.current, onSubmit };
};

const entries = () =>
  useSlashCommandInfoStore.getState().entriesByConversation[CONV] ?? [];

describe("useSkillsInterceptor", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useSlashCommandInfoStore.setState({ entriesByConversation: {} });
    mockUseConversationSkills.mockReturnValue({ data: MOCK_SKILLS });
  });

  it("passes a non-skills message straight through to onSubmit", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("hello there");
    expect(onSubmit).toHaveBeenCalledWith("hello there");
    expect(entries()).toHaveLength(0);
  });

  it("renders the loaded skills without a network call", () => {
    const { intercept, onSubmit } = setup(CONV);
    intercept("/skills");
    expect(onSubmit).not.toHaveBeenCalled();
    expect(entries()).toHaveLength(1);
    const body = entries()[0].body;
    expect(body).toContain("`init` (agentskills)");
    expect(body).toContain("Initialise a repository");
    expect(body).toContain("`releasenotes` (repo)");
  });

  it("notes when no skills are loaded", () => {
    mockUseConversationSkills.mockReturnValue({ data: [] });
    const { intercept } = setup(CONV);
    intercept("/skills");
    expect(entries()[0].body).toContain("No skills loaded");
  });

  it("passes through (no entry) when there is no conversation", () => {
    const { intercept, onSubmit } = setup(null);
    intercept("/skills");
    expect(onSubmit).toHaveBeenCalledWith("/skills");
    expect(entries()).toHaveLength(0);
  });
});
