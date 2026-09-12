import React from "react";
import { act, renderHook } from "@testing-library/react";
import { describe, it, expect, vi, afterEach } from "vitest";
import {
  type SlashCommandItem,
  useSlashCommand,
} from "#/hooks/chat/use-slash-command";
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
  content: string | undefined = `Description of ${name}`,
) {
  return { name, type, source: null, content, triggers };
}

function makeProfile(
  name: string,
  model: string | null,
): NonNullable<typeof mockLlmProfiles.data>["profiles"][number] {
  return {
    name,
    model,
    base_url: null,
    api_key_set: true,
  };
}

function makeChatInputRef() {
  const current = document.createElement("div");
  current.tabIndex = 0;
  return { current };
}

function setInputText(
  element: HTMLDivElement,
  text: string,
  cursorOffset = text.length,
) {
  element.textContent = text;
  element.innerText = text;
  if (!element.isConnected) document.body.appendChild(element);

  const textNode = element.firstChild;
  if (!textNode) return;

  const range = document.createRange();
  const selection = window.getSelection();
  range.setStart(textNode, cursorOffset);
  range.collapse(true);
  selection?.removeAllRanges();
  selection?.addRange(range);
}

function makeKeyboardEvent(key: string) {
  return {
    key,
    preventDefault: vi.fn(),
  } as unknown as React.KeyboardEvent;
}

function getSelectedCharacterOffset(element: HTMLElement) {
  const selection = window.getSelection();
  expect(selection?.rangeCount).toBe(1);
  const range = selection!.getRangeAt(0);
  const beforeCursor = range.cloneRange();
  beforeCursor.selectNodeContents(element);
  beforeCursor.setEnd(range.startContainer, range.startOffset);
  return beforeCursor.toString().length;
}

const fallbackItem: SlashCommandItem = {
  command: "/fallback",
  skill: makeSkill("fallback", ["/fallback"]),
};

const cloudBackend: Backend = {
  id: "prod",
  name: "Production",
  host: "https://app.all-hands.dev",
  apiKey: "bearer-token",
  kind: "cloud",
};

describe("useSlashCommand", () => {
  afterEach(() => {
    vi.clearAllMocks();
    mockSkills.data = undefined;
    mockSkills.isLoading = false;
    mockLlmProfiles.data = undefined;
    mockLlmProfiles.isLoading = false;
    mockConversation.data = undefined;
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
        makeProfile("haiku", "anthropic/claude-haiku-4-5"),
        makeProfile("gpt", "openai/gpt-5.1"),
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
        makeProfile("haiku", "anthropic/claude-haiku-4-5"),
        makeProfile("gpt", "openai/gpt-5.1"),
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

  it("keeps built-in commands stable while skills load or have no data", () => {
    mockSkills.isLoading = true;
    mockSkills.data = [makeSkill("hidden-until-loaded", ["/hidden"])];

    const ref = makeChatInputRef();
    const { result, rerender } = renderHook(() => useSlashCommand(ref));

    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
    ]);

    mockSkills.isLoading = false;
    mockSkills.data = undefined;
    rerender();

    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
    ]);
  });

  it("adds newly loaded skills after the hook has rendered", () => {
    mockSkills.data = [];
    const ref = makeChatInputRef();
    const { result, rerender } = renderHook(() => useSlashCommand(ref));
    expect(
      result.current.filteredItems.map((item) => item.command),
    ).not.toContain("/late-arrival");

    mockSkills.data = [makeSkill("late-arrival", ["/late-arrival"])];
    rerender();

    expect(result.current.filteredItems.map((item) => item.command)).toContain(
      "/late-arrival",
    );
  });

  it("builds commands from explicit triggers and agent skill names", () => {
    mockSkills.data = [
      makeSkill("explicit", ["task", "/explicit", "/alias"]),
      makeSkill("derived", ["task"]),
      { ...makeSkill("missing-triggers"), triggers: undefined },
      makeSkill("knowledge-hidden", [], "knowledge"),
      makeSkill("knowledge-explicit", ["/reference"], "knowledge"),
    ];

    const ref = makeChatInputRef();
    const { result } = renderHook(() => useSlashCommand(ref));

    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
      "/explicit",
      "/alias",
      "/derived",
      "/missing-triggers",
      "/reference",
    ]);
  });

  it("describes a profile without a model and filters it by name", () => {
    mockSkills.data = [];
    mockLlmProfiles.data = {
      profiles: [makeProfile("Default Profile", null)],
      active_profile: null,
    };
    const ref = makeChatInputRef();
    setInputText(ref.current, "/model DEFAULT");

    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());

    expect(result.current.filteredItems).toEqual([
      {
        command: "/model Default Profile",
        skill: {
          name: "Default Profile",
          type: "agentskills",
          source: null,
          content: "Switch to this LLM profile",
          triggers: ["/model Default Profile"],
        },
      },
    ]);
  });

  it.each([
    ["command", "/PLO", "/deploy"],
    ["skill name", "/nostic", "/inspect"],
    ["description", "/INCIDENT", "/respond"],
  ])(
    "filters skill suggestions by %s case-insensitively",
    (_field, input, command) => {
      mockSkills.data = [
        makeSkill("ship", ["/deploy"], "agentskills", "Release software"),
        makeSkill("diagnostics", ["/inspect"], "agentskills", "Check health"),
        makeSkill(
          "on-call",
          ["/respond"],
          "agentskills",
          "Fix production incidents",
        ),
        makeSkill("undocumented", ["/quiet"], "agentskills", undefined),
      ];
      const ref = makeChatInputRef();
      setInputText(ref.current, input);

      const { result } = renderHook(() => useSlashCommand(ref));
      act(() => result.current.updateSlashMenu());

      expect(result.current.filteredItems.map((item) => item.command)).toEqual([
        command,
      ]);
    },
  );

  it("filters skills without descriptions without throwing", () => {
    mockSkills.data = [
      { ...makeSkill("undocumented", ["/quiet"]), content: undefined },
    ];
    const ref = makeChatInputRef();
    setInputText(ref.current, "/absent");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(true);
    expect(result.current.filteredItems).toEqual([]);
  });

  it("recognizes a slash command after an internal newline", () => {
    mockSkills.data = [makeSkill("deploy", ["/deploy"])];
    const ref = makeChatInputRef();
    setInputText(ref.current, "context\n/dep");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(true);
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/deploy",
    ]);
  });

  it("ignores repeated trailing newlines while detecting and replacing a command", () => {
    mockSkills.data = [makeSkill("hello", ["/hello"])];
    const ref = makeChatInputRef();
    setInputText(ref.current, "/hel\n\r");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/hello",
    ]);

    act(() => result.current.selectItem(result.current.filteredItems[0]));
    expect(ref.current.textContent).toBe("/hello ");
  });

  it("removes generated trailing newlines that sit after the cursor", () => {
    mockSkills.data = [makeSkill("hello", ["/hello"])];
    const ref = makeChatInputRef();
    setInputText(ref.current, "/hel\n\r", "/hel".length);
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    act(() => result.current.selectItem(result.current.filteredItems[0]));

    expect(ref.current.textContent).toBe("/hello ");
  });

  it("closes the menu when the input has no active slash word", () => {
    mockSkills.data = [makeSkill("deploy", ["/deploy"])];
    const ref = makeChatInputRef();
    setInputText(ref.current, "/dep");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    expect(result.current.isMenuOpen).toBe(true);

    setInputText(ref.current, "/deploy ");
    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(false);
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
      "/deploy",
    ]);
  });

  it("does not open without a chat input or a cursor selection", () => {
    const missingRef: React.RefObject<HTMLDivElement | null> = {
      current: null,
    };
    const missing = renderHook(() => useSlashCommand(missingRef));

    act(() => {
      missing.result.current.updateSlashMenu();
      missing.result.current.selectItem(fallbackItem);
    });
    expect(missing.result.current.isMenuOpen).toBe(false);

    const ref = makeChatInputRef();
    setInputText(ref.current, "/btw");
    window.getSelection()?.removeAllRanges();
    const withoutSelection = renderHook(() => useSlashCommand(ref));

    act(() => withoutSelection.result.current.updateSlashMenu());
    expect(withoutSelection.result.current.isMenuOpen).toBe(false);
  });

  it("keeps model completion open while profiles load and closes when none exist", () => {
    mockSkills.data = [];
    mockLlmProfiles.isLoading = true;
    const ref = makeChatInputRef();
    setInputText(ref.current, "/model");
    const { result, rerender } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    expect(result.current.isMenuOpen).toBe(true);
    expect(result.current.filteredItems).toEqual([]);

    mockLlmProfiles.isLoading = false;
    rerender();
    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(false);
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
    ]);
  });

  it("does not treat extra model arguments as a profile completion", () => {
    mockLlmProfiles.data = {
      profiles: [makeProfile("haiku", "anthropic/claude-haiku")],
      active_profile: null,
    };
    const ref = makeChatInputRef();
    setInputText(ref.current, "/model haiku extra");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.isMenuOpen).toBe(false);
  });

  it("supports multiple spaces before a model profile filter", () => {
    mockLlmProfiles.data = {
      profiles: [
        makeProfile("haiku", "anthropic/claude-haiku"),
        makeProfile("gpt", "openai/gpt"),
      ],
      active_profile: null,
    };
    const ref = makeChatInputRef();
    setInputText(ref.current, "/model   hai");
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());

    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/model haiku",
    ]);
  });

  it("replaces the entire input when selection happens without a tracked slash range", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "draft text\n");
    const inputListener = vi.fn<(event: InputEvent) => void>();
    document.body.addEventListener("input", inputListener);
    const focus = vi.spyOn(ref.current, "focus").mockImplementation(() => {});
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.selectItem(fallbackItem));

    expect(ref.current.textContent).toBe("/fallback ");
    expect(inputListener).toHaveBeenCalledOnce();
    expect(inputListener.mock.calls[0][0].bubbles).toBe(true);
    expect(focus).toHaveBeenCalledOnce();
    expect(getSelectedCharacterOffset(ref.current)).toBe("/fallback ".length);
    expect(result.current.isMenuOpen).toBe(false);
    expect(result.current.selectedIndex).toBe(0);
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/btw",
      "/model",
      "/goal",
      "/plan",
      "/code",
    ]);
  });

  it("supports selecting a command from an empty editor", () => {
    const ref = makeChatInputRef();
    ref.current.innerText = "";
    ref.current.textContent = "";
    document.body.appendChild(ref.current);
    const range = document.createRange();
    range.setStart(ref.current, 0);
    range.collapse(true);
    window.getSelection()?.removeAllRanges();
    window.getSelection()?.addRange(range);
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    expect(result.current.isMenuOpen).toBe(false);

    act(() => result.current.selectItem(fallbackItem));
    expect(ref.current.textContent).toBe("/fallback ");
  });

  it("does not restore cleared text when selecting a previously tracked command", () => {
    mockSkills.data = [makeSkill("fallback", ["/fallback"])];
    const ref = makeChatInputRef();
    setInputText(ref.current, "/fall");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());

    ref.current.innerText = "";
    ref.current.textContent = "";
    act(() => result.current.selectItem(result.current.filteredItems[0]));

    expect(ref.current.textContent).toBe("/fallback ");
  });

  it("replaces only the slash word when the cursor is in its middle", () => {
    mockSkills.data = [makeSkill("hello", ["/hello"])];
    const ref = makeChatInputRef();
    const input = "prefix\n/helXYZ suffix";
    setInputText(ref.current, input, input.indexOf("/hel") + 4);
    const inputListener = vi.fn();
    ref.current.addEventListener("input", inputListener);
    const focus = vi.spyOn(ref.current, "focus").mockImplementation(() => {});
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/hello",
    ]);

    act(() => result.current.selectItem(result.current.filteredItems[0]));

    expect(ref.current.textContent).toBe("prefix\n/hello  suffix");
    expect(inputListener).toHaveBeenCalledOnce();
    expect(focus).toHaveBeenCalledOnce();
    expect(getSelectedCharacterOffset(ref.current)).toBe(
      "prefix\n".length + "/hello ".length,
    );
    expect(result.current.isMenuOpen).toBe(false);
  });

  it("replaces a model profile token through its trailing characters", () => {
    mockSkills.data = [];
    mockLlmProfiles.data = {
      profiles: [makeProfile("haiku", "anthropic/claude-haiku")],
      active_profile: null,
    };
    const ref = makeChatInputRef();
    const input = "question /model haiXYZ later";
    setInputText(ref.current, input, input.indexOf("hai") + 3);
    const { result } = renderHook(() => useSlashCommand(ref));

    act(() => result.current.updateSlashMenu());
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/model haiku",
    ]);

    act(() => result.current.selectItem(result.current.filteredItems[0]));
    expect(ref.current.textContent).toBe("question /model haiku  later");
  });

  it("uses the latest input ref for direct selection after rerendering", () => {
    mockSkills.data = [makeSkill("fallback", ["/fallback"])];
    const firstRef = makeChatInputRef();
    setInputText(firstRef.current, "first draft");
    const secondRef = makeChatInputRef();
    setInputText(secondRef.current, "/fall");
    const { result, rerender } = renderHook(
      ({ inputRef }) => useSlashCommand(inputRef),
      { initialProps: { inputRef: firstRef } },
    );

    rerender({ inputRef: secondRef });
    act(() => result.current.updateSlashMenu());
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/fallback",
    ]);

    act(() => result.current.selectItem(result.current.filteredItems[0]));
    expect(secondRef.current.textContent).toBe("/fallback ");
    expect(firstRef.current.textContent).toBe("first draft");
  });

  it("uses the latest selection callback for keyboard completion after rerendering", () => {
    mockSkills.data = [makeSkill("fallback", ["/fallback"])];
    const firstRef = makeChatInputRef();
    setInputText(firstRef.current, "first draft");
    const secondRef = makeChatInputRef();
    setInputText(secondRef.current, "/fall");
    const { result, rerender } = renderHook(
      ({ inputRef }) => useSlashCommand(inputRef),
      { initialProps: { inputRef: firstRef } },
    );

    rerender({ inputRef: secondRef });
    act(() => result.current.updateSlashMenu());
    act(() => result.current.handleSlashKeyDown(makeKeyboardEvent("Enter")));

    expect(secondRef.current.textContent).toBe("/fallback ");
    expect(firstRef.current.textContent).toBe("first draft");
  });

  it("still replaces text when the browser has no active Selection object", () => {
    mockSkills.data = [makeSkill("fallback", ["/fallback"])];
    const trackedRef = makeChatInputRef();
    setInputText(trackedRef.current, "/fall");
    const tracked = renderHook(() => useSlashCommand(trackedRef));
    act(() => tracked.result.current.updateSlashMenu());

    const trackedSelection = vi
      .spyOn(window, "getSelection")
      .mockReturnValue(null);
    act(() =>
      tracked.result.current.selectItem(
        tracked.result.current.filteredItems[0],
      ),
    );
    expect(trackedRef.current.textContent).toBe("/fallback ");
    trackedSelection.mockRestore();

    const fallbackRef = makeChatInputRef();
    setInputText(fallbackRef.current, "draft");
    const fallback = renderHook(() => useSlashCommand(fallbackRef));
    const fallbackSelection = vi
      .spyOn(window, "getSelection")
      .mockReturnValue(null);
    act(() => fallback.result.current.selectItem(fallbackItem));
    expect(fallbackRef.current.textContent).toBe("/fallback ");
    fallbackSelection.mockRestore();
  });

  it("resets keyboard selection when the slash filter changes", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());

    const down = makeKeyboardEvent("ArrowDown");
    act(() => result.current.handleSlashKeyDown(down));
    expect(result.current.selectedIndex).toBe(1);

    setInputText(ref.current, "/go");
    act(() => result.current.updateSlashMenu());
    expect(result.current.selectedIndex).toBe(0);
    expect(result.current.filteredItems.map((item) => item.command)).toEqual([
      "/goal",
    ]);
  });

  it("wraps keyboard selection in both directions", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());

    for (const expected of [1, 2, 3, 4, 0]) {
      const event = makeKeyboardEvent("ArrowDown");
      act(() => expect(result.current.handleSlashKeyDown(event)).toBe(true));
      expect(event.preventDefault).toHaveBeenCalledOnce();
      expect(result.current.selectedIndex).toBe(expected);
    }

    for (const expected of [4, 3]) {
      const event = makeKeyboardEvent("ArrowUp");
      act(() => expect(result.current.handleSlashKeyDown(event)).toBe(true));
      expect(event.preventDefault).toHaveBeenCalledOnce();
      expect(result.current.selectedIndex).toBe(expected);
    }
  });

  it.each([
    ["Enter", "/btw "],
    ["Tab", "/btw "],
  ])("selects the highlighted command with %s", (key, replacement) => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());
    const event = makeKeyboardEvent(key);

    act(() => expect(result.current.handleSlashKeyDown(event)).toBe(true));

    expect(event.preventDefault).toHaveBeenCalledOnce();
    expect(ref.current.textContent).toBe(replacement);
    expect(result.current.isMenuOpen).toBe(false);
  });

  it("does not consume selection when a changing profile list invalidates the index", () => {
    mockLlmProfiles.data = {
      profiles: [
        makeProfile("one", "model-1"),
        makeProfile("two", "model-2"),
        makeProfile("three", "model-3"),
      ],
      active_profile: null,
    };
    const ref = makeChatInputRef();
    setInputText(ref.current, "/model");
    const { result, rerender } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());
    act(() => result.current.handleSlashKeyDown(makeKeyboardEvent("ArrowUp")));
    expect(result.current.selectedIndex).toBe(2);

    mockLlmProfiles.data = {
      profiles: [makeProfile("one", "model-1")],
      active_profile: null,
    };
    rerender();
    const enter = makeKeyboardEvent("Enter");

    act(() => expect(result.current.handleSlashKeyDown(enter)).toBe(false));
    expect(enter.preventDefault).not.toHaveBeenCalled();
    expect(ref.current.textContent).toBe("/model");
  });

  it("ignores keyboard input while closed or while filtering has no results", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    const closed = makeKeyboardEvent("ArrowDown");
    expect(result.current.handleSlashKeyDown(closed)).toBe(false);
    expect(closed.preventDefault).not.toHaveBeenCalled();

    setInputText(ref.current, "/nothing-matches");
    act(() => result.current.updateSlashMenu());
    expect(result.current.isMenuOpen).toBe(true);
    expect(result.current.filteredItems).toEqual([]);
    const empty = makeKeyboardEvent("ArrowDown");
    expect(result.current.handleSlashKeyDown(empty)).toBe(false);
    expect(empty.preventDefault).not.toHaveBeenCalled();
  });

  it("closes and consumes Escape", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());
    const escape = makeKeyboardEvent("Escape");

    act(() => expect(result.current.handleSlashKeyDown(escape)).toBe(true));

    expect(escape.preventDefault).toHaveBeenCalledOnce();
    expect(result.current.isMenuOpen).toBe(false);
  });

  it.each(["ArrowLeft", "ArrowRight", "Home", "End"])(
    "closes without consuming %s",
    (key) => {
      const ref = makeChatInputRef();
      setInputText(ref.current, "/");
      const { result } = renderHook(() => useSlashCommand(ref));
      act(() => result.current.updateSlashMenu());
      const event = makeKeyboardEvent(key);

      act(() => expect(result.current.handleSlashKeyDown(event)).toBe(false));

      expect(event.preventDefault).not.toHaveBeenCalled();
      expect(result.current.isMenuOpen).toBe(false);
    },
  );

  it("leaves the menu open for an unrelated key", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());
    const event = makeKeyboardEvent("Shift");

    expect(result.current.handleSlashKeyDown(event)).toBe(false);
    expect(event.preventDefault).not.toHaveBeenCalled();
    expect(result.current.isMenuOpen).toBe(true);
  });

  it("allows callers to close the menu", () => {
    const ref = makeChatInputRef();
    setInputText(ref.current, "/");
    const { result } = renderHook(() => useSlashCommand(ref));
    act(() => result.current.updateSlashMenu());

    act(() => result.current.closeMenu());

    expect(result.current.isMenuOpen).toBe(false);
  });
});
