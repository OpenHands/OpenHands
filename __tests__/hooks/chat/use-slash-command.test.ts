import React from "react";
import { act, renderHook } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { useSlashCommand } from "#/hooks/chat/use-slash-command";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import type { Backend } from "#/api/backend-registry/types";

const mockSkills = vi.hoisted(() => ({
  data: undefined as unknown[] | undefined,
  isLoading: false,
}));

const mockConversation = vi.hoisted(() => ({
  data: undefined as { conversation_version?: "V0" | "V1" } | undefined,
}));

vi.mock("#/hooks/query/use-skills", () => ({
  useSkills: () => mockSkills,
}));

const mockLlmProfiles = vi.hoisted(() => ({
  data: undefined as
    | {
        profiles: Array<{
          name: string;
          model: string | null;
          base_url: string | null;
          api_key_set: boolean;
        }>;
        active_profile: string | null;
      }
    | undefined,
  isLoading: false,
}));

vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => mockSkills,
}));

vi.mock("#/hooks/query/use-llm-profiles", () => ({
  useLlmProfiles: () => mockLlmProfiles,
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

function setInputText(element: HTMLDivElement, text: string) {
  element.textContent = text;
  element.innerText = text;
  document.body.appendChild(element);

  const textNode = element.firstChild;
  if (!textNode) return;

  const range = document.createRange();
  const selection = window.getSelection();
  range.setStart(textNode, text.length);
  range.collapse(true);
  selection?.removeAllRanges();
  selection?.addRange(range);
}

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

describe("useSlashCommand", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSkills.data = undefined;
    mockSkills.isLoading = false;
    mockLlmProfiles.data = undefined;
    mockLlmProfiles.isLoading = false;
    mockConversation.data = undefined;
  });

  afterEach(() => {
    document.body.innerHTML = "";
    window.localStorage.clear?.();
    __resetActiveStoreForTests();
  });

  it("excludes /new from the built-in commands on a local backend", () => {
    // Arrange — default active backend is the bundled local one.
    mockConversation.data = { conversation_version: "V1" };
    mockSkills.data = [makeSkill("code-search", ["/code-search"])];

    // Act
    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    // Assert
    const commands = result.current.filteredItems.map((i) => i.command);
    expect(commands).not.toContain("/new");
    expect(commands).toEqual(expect.arrayContaining(["/btw", "/code-search"]));
  });

  it("includes /new in the built-in commands on a cloud backend", () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    mockConversation.data = { conversation_version: "V1" };
    mockSkills.data = [];

    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(ActiveBackendProvider, null, children);

    // Act
    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref), { wrapper });

    // Assert
    const commands = result.current.filteredItems.map((i) => i.command);
    expect(commands).toContain("/new");
  });

  it("suggests saved LLM profiles after /model on a local backend", () => {
    // The active backend store is reset before each test, which restores the default local backend.

    mockSkills.data = [];
    mockLlmProfiles.data = {
      profiles: [
        {
          name: "haiku",
          model: "anthropic/claude-haiku-4-5",
          base_url: null,
          api_key_set: true,
        },
        {
          name: "gpt",
          model: "openai/gpt-5.1",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: "haiku",
    };

    const ref = makeChatInputRef();
    setInputText(ref.current, "/model");

    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(true);
    expect(result.current.filteredItems.map((i) => i.command)).toEqual([
      "/model haiku",
      "/model gpt",
    ]);
  });

  it("filters saved LLM profile suggestions by profile name or model", () => {
    mockSkills.data = [];
    mockLlmProfiles.data = {
      profiles: [
        {
          name: "haiku",
          model: "anthropic/claude-haiku-4-5",
          base_url: null,
          api_key_set: true,
        },
        {
          name: "gpt",
          model: "openai/gpt-5.1",
          base_url: null,
          api_key_set: true,
        },
      ],
      active_profile: null,
    };

    const ref = makeChatInputRef();
    setInputText(ref.current, "/model claude");

    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.filteredItems.map((i) => i.command)).toEqual([
      "/model haiku",
    ]);
  });
});
