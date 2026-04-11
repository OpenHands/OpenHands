import { renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { useSlashCommand } from "#/hooks/chat/use-slash-command";

const mockSkills = vi.hoisted(() => ({
  data: undefined as unknown[] | undefined,
  isLoading: false,
}));

const mockConversation = vi.hoisted(() => ({
  data: undefined as { conversation_version?: "V0" | "V1" } | undefined,
}));

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => mockSkills,
}));

vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => mockConversation,
}));

function makeSkill(
  name: string,
  triggers: string[] = [],
  type: "agentskills" | "knowledge" = "agentskills",
) {
  return { name, type, content: `Description of ${name}`, triggers };
}

function makeChatInputRef() {
  return { current: document.createElement("div") };
}

describe("useSlashCommand", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSkills.data = undefined;
    mockSkills.isLoading = false;
    mockConversation.data = undefined;
  });

  it("includes /new built-in command for V1 conversations", () => {
    mockConversation.data = { conversation_version: "V1" };
    mockSkills.isLoading = false;
    mockSkills.data = [makeSkill("code-search", ["/code-search"])];

    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    const commands = result.current.filteredItems.map((i) => i.command);
    expect(commands).toContain("/new");
    expect(commands).toContain("/code-search");
  });
});
