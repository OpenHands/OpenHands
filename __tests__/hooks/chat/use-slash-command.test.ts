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

vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  const definitions = await import("#/i18n/translation.json");
  const translations = definitions.default as Record<
    string,
    Record<string, string>
  >;
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => translations[key]?.de ?? key,
      i18n: { language: "de", exists: () => true },
    }),
  };
});

const mockSkills = vi.hoisted(() => ({
  data: undefined as unknown[] | undefined,
  isLoading: false,
}));

const mockConversation = vi.hoisted(() => ({
  data: undefined as
    | {
        id: string;
        conversation_version?: "V0" | "V1";
        agent_kind?: "openhands" | "acp" | null;
        supports_manual_condensation?: boolean;
      }
    | undefined,
}));

const mockRoute = vi.hoisted(() => ({ conversationId: null as string | null }));

vi.mock("#/hooks/use-conversation-id", () => ({
  useOptionalConversationId: () => ({
    conversationId: mockRoute.conversationId,
  }),
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
  const target = element;
  target.textContent = text;
  target.innerText = text;
  document.body.appendChild(target);

  const textNode = target.firstChild;
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
    mockRoute.conversationId = null;
  });

  afterEach(() => {
    document.body.innerHTML = "";
    window.localStorage.clear?.();
    __resetActiveStoreForTests();
  });

  it("includes local conversation commands on a local backend", () => {
    // Arrange — default active backend is the bundled local one.
    mockConversation.data = {
      id: "local-conversation",
      conversation_version: "V1",
      agent_kind: "openhands",
      supports_manual_condensation: true,
    };
    mockRoute.conversationId = "local-conversation";
    mockSkills.data = [makeSkill("code-search", ["/code-search"])];

    // Act
    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    // Assert
    const commands = result.current.filteredItems.map((i) => i.command);
    expect(commands).toEqual(
      expect.arrayContaining([
        "/new",
        "/confirm",
        "/condense",
        "/fork",
        "/btw",
        "/code-search",
      ]),
    );
  });

  it("excludes commands without Cloud API support on a cloud backend", () => {
    // Arrange
    setRegisteredBackends([cloudBackend]);
    setActiveSelection({ backendId: cloudBackend.id });
    mockConversation.data = {
      id: "cloud-conversation",
      conversation_version: "V1",
    };
    mockRoute.conversationId = "cloud-conversation";
    mockSkills.data = [];

    const wrapper = ({ children }: { children: React.ReactNode }) =>
      React.createElement(ActiveBackendProvider, null, children);

    // Act
    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref), { wrapper });

    // Assert
    const commands = result.current.filteredItems.map((i) => i.command);
    expect(commands).toContain("/help");
    expect(commands).toContain("/new");
    expect(commands).toContain("/confirm");
    expect(commands).not.toContain("/condense");
    expect(commands).not.toContain("/fork");
  });

  it("omits /confirm from local ACP conversations", () => {
    mockConversation.data = {
      id: "local-acp-conversation",
      conversation_version: "V1",
      agent_kind: "acp",
      supports_manual_condensation: false,
    };
    mockRoute.conversationId = "local-acp-conversation";
    mockSkills.data = [];

    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    expect(result.current.filteredItems.map((i) => i.command)).not.toContain(
      "/confirm",
    );
    expect(result.current.filteredItems.map((i) => i.command)).not.toContain(
      "/condense",
    );
  });

  it("omits /condense when the local OpenHands condenser is incompatible", () => {
    mockConversation.data = {
      id: "local-conversation",
      conversation_version: "V1",
      agent_kind: "openhands",
      supports_manual_condensation: false,
    };
    mockRoute.conversationId = "local-conversation";
    mockSkills.data = [];

    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    expect(result.current.filteredItems.map((i) => i.command)).not.toContain(
      "/condense",
    );
  });

  it("only includes context-free built-ins and skill commands before a conversation", () => {
    mockSkills.data = [makeSkill("code-search", ["/code-search"])];

    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    expect(result.current.filteredItems.map((i) => i.command)).toEqual([
      "/help",
      "/history",
      "/settings",
      "/skills",
      "/feedback",
      "/model",
      "/code-search",
    ]);
  });

  it("does not treat a task route as a running conversation", () => {
    mockRoute.conversationId = "task-abc";
    mockConversation.data = undefined;
    mockSkills.data = [];

    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));
    const commands = result.current.filteredItems.map((item) => item.command);

    expect(commands).not.toContain("/new");
    expect(commands).not.toContain("/confirm");
    expect(commands).not.toContain("/condense");
    expect(commands).not.toContain("/fork");
    expect(commands).not.toContain("/goal");
    expect(commands).not.toContain("/btw");
  });

  it("matches /help despite contentEditable zero-width formatting characters", () => {
    mockSkills.data = [];
    const ref = makeChatInputRef();
    setInputText(ref.current, "\u200B/he\u200Blp");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(true);
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/help",
    ]);
  });

  it("filters built-ins by their localized displayed description", () => {
    mockSkills.data = [];
    const ref = makeChatInputRef();
    setInputText(ref.current, "/verfügbare");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/help",
    ]);
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
