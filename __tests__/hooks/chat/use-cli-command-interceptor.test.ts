import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import {
  SKILLS_COMMAND_DEADLINE_MS,
  useCliCommandInterceptor,
  useUiCommandInterceptor,
} from "#/hooks/chat/use-cli-command-interceptor";
import { normalizeUiCommand } from "#/utils/slash-command-text";
import { buildSlashCommandCatalog } from "#/utils/slash-command-catalog";

const mocks = vi.hoisted(() => ({
  backendKind: "local" as "local" | "cloud",
  isMobile: false,
  navigate: vi.fn(),
  toggleDesktopSidebar: vi.fn(),
  toggleMobileSidebar: vi.fn(),
  beginSkills: vi.fn(),
  completeSkills: vi.fn(),
  failSkills: vi.fn(),
  deactivateSkillsPlacementFallback: vi.fn(),
  showSkills: vi.fn(),
  showHelp: vi.fn(),
  reserveInvocationOrder: vi.fn(),
  refetchSkills: vi.fn(),
  forkConversation: vi.fn(),
  condenseConversation: vi.fn(),
  getLoadedResources: vi.fn(),
  displayErrorToast: vi.fn(),
  displayLoadingToast: vi.fn(() => "toast-1"),
  dismissToast: vi.fn(),
  displaySuccessToast: vi.fn(),
  displayWarningToast: vi.fn(),
  openConfirmationPolicy: vi.fn(),
  timelineBoundaryEventId: "event-7",
  skills: undefined as
    | Array<{
        name: string;
        type: "agentskills";
        source: string;
        description: string;
      }>
    | undefined,
  conversation: {
    id: "conversation-1",
    title: "Investigate bug",
    agent_kind: "openhands" as "openhands" | "acp",
    supports_manual_condensation: true,
    conversation_url: "http://runtime.example",
    session_api_key: "runtime-key",
  },
  loadedResources: {
    skills: [
      {
        name: "review",
        source: "project",
        description: "Review code",
      },
    ],
    hooks: [{ hookType: "pre_tool_use", commands: ["lint"] }],
    mcps: [{ name: "github", transport: "stdio" }],
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { kind: mocks.backendKind },
  }),
}));
vi.mock("#/context/navigation-context", () => ({
  useNavigation: () => ({ navigate: mocks.navigate }),
}));
vi.mock("#/hooks/use-breakpoint", () => ({
  SIDEBAR_RAIL_COLLAPSE_MAX_WIDTH: 767,
  useBreakpoint: () => mocks.isMobile,
}));
vi.mock("#/stores/sidebar-store", () => ({
  useSidebarStore: (
    selector: (state: { toggleCollapsed: () => void }) => unknown,
  ) => selector({ toggleCollapsed: mocks.toggleDesktopSidebar }),
}));
vi.mock("#/components/features/sidebar/sidebar-mobile-nav-context", () => ({
  useOptionalSidebarMobileNav: () => ({
    toggle: mocks.toggleMobileSidebar,
  }),
}));
vi.mock("#/hooks/query/use-active-conversation", () => ({
  useActiveConversation: () => ({ data: mocks.conversation }),
}));
vi.mock("#/hooks/query/use-conversation-skills", () => ({
  useConversationSkills: () => ({
    data: mocks.skills,
    refetch: mocks.refetchSkills,
  }),
}));
vi.mock("#/stores/slash-command-output-store", () => ({
  useSlashCommandOutputStore: (
    selector: (state: {
      beginSkills: typeof mocks.beginSkills;
      completeSkills: typeof mocks.completeSkills;
      failSkills: typeof mocks.failSkills;
      deactivateSkillsPlacementFallback: typeof mocks.deactivateSkillsPlacementFallback;
      showSkills: typeof mocks.showSkills;
      showHelp: typeof mocks.showHelp;
      reserveInvocationOrder: typeof mocks.reserveInvocationOrder;
    }) => unknown,
  ) =>
    selector({
      beginSkills: mocks.beginSkills,
      completeSkills: mocks.completeSkills,
      failSkills: mocks.failSkills,
      deactivateSkillsPlacementFallback:
        mocks.deactivateSkillsPlacementFallback,
      showSkills: mocks.showSkills,
      showHelp: mocks.showHelp,
      reserveInvocationOrder: mocks.reserveInvocationOrder,
    }),
}));
vi.mock("#/hooks/mutation/use-fork-conversation", () => ({
  useForkConversation: () => ({ mutate: mocks.forkConversation }),
}));
vi.mock("#/hooks/chat/slash-command-timeline-boundary", () => ({
  getLastConversationTimelineEventId: () => mocks.timelineBoundaryEventId,
}));
vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      condenseConversation: mocks.condenseConversation,
      getLoadedResources: mocks.getLoadedResources,
    },
  }),
);
vi.mock("#/utils/custom-toast-handlers", () => ({
  dismissToast: mocks.dismissToast,
  displayErrorToast: mocks.displayErrorToast,
  displayLoadingToast: mocks.displayLoadingToast,
  displaySuccessToast: mocks.displaySuccessToast,
  displayWarningToast: mocks.displayWarningToast,
}));
vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  const definitions = await import("#/i18n/translation.json");
  const translations = definitions.default as Record<
    string,
    Record<string, string>
  >;
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, options?: Record<string, unknown>) =>
        (translations[key]?.de ?? key).replace(
          /\{\{(\w+)\}\}/g,
          (placeholder, name: string) =>
            options?.[name] === undefined ? placeholder : String(options[name]),
        ),
    }),
  };
});

describe("useCliCommandInterceptor", () => {
  afterEach(() => {
    vi.useRealTimers();
  });

  beforeEach(() => {
    vi.clearAllMocks();
    mocks.backendKind = "local";
    mocks.isMobile = false;
    mocks.conversation.agent_kind = "openhands";
    mocks.conversation.supports_manual_condensation = true;
    mocks.timelineBoundaryEventId = "event-7";
    mocks.skills = [
      {
        name: "review",
        type: "agentskills",
        source: "project",
        description: "Review code",
      },
    ];
    mocks.refetchSkills.mockResolvedValue({ data: mocks.skills });
    mocks.getLoadedResources.mockResolvedValue(mocks.loadedResources);
    mocks.condenseConversation.mockResolvedValue(undefined);
    mocks.beginSkills.mockReturnValue("skills-entry-1");
    let invocationOrder = 0;
    mocks.reserveInvocationOrder.mockImplementation(() => invocationOrder++);
  });

  const setup = (conversationId: string | null = "conversation-1") => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useCliCommandInterceptor(conversationId, onSubmit, {
        onOpenConfirmationPolicy: mocks.openConfirmationPolicy,
      }),
    );
    return { submit: result.current, onSubmit };
  };

  const setupHome = () => {
    const onSubmit = vi.fn();
    const { result } = renderHook(() =>
      useUiCommandInterceptor(onSubmit, { outputScopeId: "home" }),
    );
    return { submit: result.current, onSubmit };
  };

  it("passes non-CLI commands through unchanged", () => {
    const { submit, onSubmit } = setup();

    act(() => submit("  ordinary prompt  "));

    expect(onSubmit).toHaveBeenCalledWith("  ordinary prompt  ");
  });

  it("toggles the desktop rail or mobile drawer", () => {
    const desktop = setup();
    act(() => desktop.submit(" /history "));
    expect(mocks.toggleDesktopSidebar).toHaveBeenCalledOnce();

    mocks.isMobile = true;
    const mobile = setup();
    act(() => mobile.submit("/history"));
    expect(mocks.toggleMobileSidebar).toHaveBeenCalledOnce();
  });

  it("routes /settings while unknown commands follow the normal submission path", () => {
    const { submit, onSubmit } = setup(null);

    act(() => submit("/settings"));
    act(() => submit("/unknown"));

    expect(mocks.navigate).toHaveBeenCalledOnce();
    expect(mocks.navigate).toHaveBeenCalledWith("/settings");
    expect(onSubmit).toHaveBeenCalledWith("/unknown");
  });

  it("opens the CLI feedback form without sending a chat message", () => {
    const open = vi.spyOn(window, "open").mockImplementation(() => null);
    const { submit, onSubmit } = setup(null);

    act(() => submit("/feedback"));

    expect(open).toHaveBeenCalledWith(
      "https://forms.gle/chHc5VdS3wty5DwW6",
      "_blank",
      "noopener,noreferrer",
    );
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("opens live confirmation policy controls for a local OpenHands conversation", () => {
    const { submit, onSubmit } = setup();

    act(() => submit("/confirm"));

    expect(mocks.openConfirmationPolicy).toHaveBeenCalledOnce();
    expect(mocks.displayWarningToast).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("consumes Cloud /confirm with an unsupported warning", () => {
    mocks.backendKind = "cloud";
    const { submit, onSubmit } = setup();

    act(() => submit("/confirm"));

    expect(mocks.openConfirmationPolicy).not.toHaveBeenCalled();
    expect(mocks.displayWarningToast).toHaveBeenCalledOnce();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("does not offer live confirmation controls to ACP conversations", () => {
    mocks.conversation.agent_kind = "acp";
    const { submit, onSubmit } = setup();

    act(() => submit("/confirm"));

    expect(mocks.openConfirmationPolicy).not.toHaveBeenCalled();
    expect(mocks.displayErrorToast).toHaveBeenCalledOnce();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("condenses a compatible active local conversation", async () => {
    const { submit, onSubmit } = setup();

    act(() => submit("/condense"));

    expect(mocks.condenseConversation).toHaveBeenCalledWith(
      "conversation-1",
      "http://runtime.example",
      "runtime-key",
    );
    expect(mocks.displayLoadingToast).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(mocks.displaySuccessToast).toHaveBeenCalledOnce(),
    );
    expect(mocks.dismissToast).toHaveBeenCalledWith("toast-1");
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("guards a conversation against duplicate condensation requests", async () => {
    let resolveCondensation!: () => void;
    mocks.condenseConversation.mockReturnValue(
      new Promise<void>((resolve) => {
        resolveCondensation = resolve;
      }),
    );
    const { submit } = setup();

    act(() => submit("/condense"));
    act(() => submit("/condense"));

    expect(mocks.condenseConversation).toHaveBeenCalledOnce();
    expect(mocks.displayLoadingToast).toHaveBeenCalledOnce();

    await act(async () => resolveCondensation());
    expect(mocks.dismissToast).toHaveBeenCalledWith("toast-1");
    expect(mocks.displaySuccessToast).toHaveBeenCalledOnce();

    await act(async () => submit("/condense"));
    expect(mocks.condenseConversation).toHaveBeenCalledTimes(2);
    expect(mocks.displayLoadingToast).toHaveBeenCalledTimes(2);
    expect(mocks.displaySuccessToast).toHaveBeenCalledTimes(2);
  });

  it("replaces condensation loading feedback with the request error", async () => {
    mocks.condenseConversation.mockRejectedValue(
      new Error("Condensation failed"),
    );
    const { submit } = setup();

    act(() => submit("/condense"));

    expect(mocks.displayLoadingToast).toHaveBeenCalledOnce();
    await waitFor(() =>
      expect(mocks.displayErrorToast).toHaveBeenCalledWith(
        "Condensation failed",
      ),
    );
    expect(mocks.dismissToast).toHaveBeenCalledWith("toast-1");
    expect(mocks.displaySuccessToast).not.toHaveBeenCalled();
  });

  it("does not invoke condensation when the running local agent is incompatible", () => {
    mocks.conversation.supports_manual_condensation = false;
    const { submit, onSubmit } = setup();

    act(() => submit("/condense"));

    expect(mocks.condenseConversation).not.toHaveBeenCalled();
    expect(mocks.displayLoadingToast).not.toHaveBeenCalled();
    expect(mocks.displayErrorToast).toHaveBeenCalledOnce();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("consumes Cloud /condense with an unsupported warning", () => {
    mocks.backendKind = "cloud";
    const { submit, onSubmit } = setup();

    act(() => submit("/condense"));

    expect(mocks.condenseConversation).not.toHaveBeenCalled();
    expect(mocks.displayLoadingToast).not.toHaveBeenCalled();
    expect(mocks.displayWarningToast).toHaveBeenCalledOnce();
    expect(mocks.displaySuccessToast).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("keeps forking local-only", () => {
    mocks.backendKind = "cloud";
    const { submit } = setup();

    act(() => submit("/fork"));

    expect(mocks.forkConversation).not.toHaveBeenCalled();
    expect(mocks.displayErrorToast).toHaveBeenCalledOnce();
  });

  it("keeps /condense and /fork out of Cloud help", () => {
    mocks.backendKind = "cloud";
    const { submit } = setup();

    act(() => submit("/help"));

    const helpItems = mocks.showHelp.mock.calls[0]?.[2] as Array<{
      command: string;
    }>;
    expect(helpItems.map((item) => item.command)).not.toContain("/fork");
    expect(helpItems.map((item) => item.command)).toContain("/confirm");
    expect(helpItems.map((item) => item.command)).not.toContain("/condense");
  });

  it("omits /condense from local help when the running agent is incompatible", () => {
    mocks.conversation.supports_manual_condensation = false;
    const { submit } = setup();

    act(() => submit("/help"));

    const helpItems = mocks.showHelp.mock.calls[0]?.[2] as Array<{
      command: string;
    }>;
    expect(helpItems.map((item) => item.command)).not.toContain("/condense");
  });

  it("consumes repeated /help submissions, including contentEditable zero-width characters", () => {
    const { submit, onSubmit } = setup();

    act(() => submit("/help"));
    mocks.timelineBoundaryEventId = "event-8";
    act(() => submit("/help"));
    mocks.timelineBoundaryEventId = "event-9";
    act(() => submit("/he\u200Blp\uFEFF"));

    expect(mocks.showHelp).toHaveBeenCalledTimes(3);
    expect(mocks.showHelp.mock.calls.map((call) => call[1])).toEqual([
      "event-7",
      "event-8",
      "event-9",
    ]);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("renders the resources serialized on the active local conversation", async () => {
    mocks.skills![0] = {
      name: "added-after-start",
      type: "agentskills",
      source: "project",
      description: "Available now, but not loaded into this conversation",
    };
    const { submit } = setup();

    act(() => submit("/skills"));

    expect(mocks.getLoadedResources).toHaveBeenCalledWith(
      "conversation-1",
      "http://runtime.example",
      "runtime-key",
    );
    expect(mocks.beginSkills).toHaveBeenCalledWith(
      "conversation-1",
      "event-7",
      expect.any(Number),
    );
    expect(mocks.refetchSkills).not.toHaveBeenCalled();
    await waitFor(() =>
      expect(mocks.completeSkills).toHaveBeenCalledWith(
        "conversation-1",
        "skills-entry-1",
        mocks.loadedResources,
      ),
    );
    expect(mocks.showSkills).not.toHaveBeenCalledWith(
      expect.anything(),
      expect.anything(),
      expect.objectContaining({
        skills: expect.arrayContaining([
          expect.objectContaining({ name: "added-after-start" }),
        ]),
      }),
    );
  });

  it("renders loaded Cloud resources from the shared conversation service", async () => {
    mocks.backendKind = "cloud";
    const cloudResources = {
      skills: [{ name: "review", source: null, description: "Review code" }],
      hooks: [{ hookType: "pre_tool_use", commands: ["lint"] }],
      mcps: null,
    };
    mocks.getLoadedResources.mockResolvedValue(cloudResources);
    const { submit } = setup();

    act(() => submit("/skills"));

    expect(mocks.getLoadedResources).toHaveBeenCalledWith(
      "conversation-1",
      "http://runtime.example",
      "runtime-key",
    );
    await waitFor(() =>
      expect(mocks.completeSkills).toHaveBeenCalledWith(
        "conversation-1",
        "skills-entry-1",
        cloudResources,
      ),
    );
    expect(mocks.displayErrorToast).not.toHaveBeenCalled();
  });

  it("records a loaded-skills failure inline without rendering an empty result", async () => {
    mocks.backendKind = "cloud";
    mocks.getLoadedResources.mockRejectedValue(
      new Error("Loaded skills unavailable"),
    );
    const { submit, onSubmit } = setup();

    act(() => submit("/skills"));

    await waitFor(() =>
      expect(mocks.failSkills).toHaveBeenCalledWith(
        "conversation-1",
        "skills-entry-1",
        "request",
      ),
    );
    expect(mocks.displayErrorToast).not.toHaveBeenCalled();
    expect(mocks.showSkills).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("anchors deferred /skills output when the command is invoked", async () => {
    let resolveResources!: (value: typeof mocks.loadedResources) => void;
    mocks.getLoadedResources.mockReturnValue(
      new Promise((resolve) => {
        resolveResources = resolve;
      }),
    );
    const { submit } = setup();

    act(() => submit("/skills"));
    mocks.timelineBoundaryEventId = "event-8";
    act(() => resolveResources(mocks.loadedResources));

    await waitFor(() =>
      expect(mocks.completeSkills).toHaveBeenCalledWith(
        "conversation-1",
        "skills-entry-1",
        mocks.loadedResources,
      ),
    );
  });

  it("inserts loading synchronously for an idle request that never settles", () => {
    mocks.getLoadedResources.mockReturnValue(new Promise(() => {}));
    const { submit, onSubmit } = setup();

    act(() => submit("/skills"));

    expect(mocks.beginSkills).toHaveBeenCalledWith(
      "conversation-1",
      "event-7",
      expect.any(Number),
    );
    expect(mocks.completeSkills).not.toHaveBeenCalled();
    expect(mocks.failSkills).not.toHaveBeenCalled();
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("turns a never-settling request into an inline timeout at the command deadline", async () => {
    vi.useFakeTimers();
    mocks.getLoadedResources.mockReturnValue(new Promise(() => {}));
    const { submit } = setup();

    act(() => submit("/skills"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(SKILLS_COMMAND_DEADLINE_MS);
    });

    expect(mocks.failSkills).toHaveBeenCalledWith(
      "conversation-1",
      "skills-entry-1",
      "timeout",
    );
    vi.useRealTimers();
  });

  it("ignores a late success after the command deadline", async () => {
    vi.useFakeTimers();
    let resolveResources!: (value: typeof mocks.loadedResources) => void;
    mocks.getLoadedResources.mockReturnValue(
      new Promise((resolve) => {
        resolveResources = resolve;
      }),
    );
    const { submit } = setup();

    act(() => submit("/skills"));
    await act(async () => {
      await vi.advanceTimersByTimeAsync(SKILLS_COMMAND_DEADLINE_MS);
    });
    act(() => resolveResources(mocks.loadedResources));
    await act(async () => Promise.resolve());

    expect(mocks.failSkills).toHaveBeenCalledOnce();
    expect(mocks.completeSkills).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("keeps a successful retry independent while the first request later times out", async () => {
    vi.useFakeTimers();
    mocks.beginSkills
      .mockReturnValueOnce("skills-entry-1")
      .mockReturnValueOnce("skills-entry-2");
    mocks.getLoadedResources
      .mockReturnValueOnce(new Promise(() => {}))
      .mockResolvedValueOnce(mocks.loadedResources);
    const { submit, onSubmit } = setup();

    act(() => submit("/skills"));
    act(() => submit("/skills"));
    await act(async () => Promise.resolve());

    expect(mocks.completeSkills).toHaveBeenCalledWith(
      "conversation-1",
      "skills-entry-2",
      mocks.loadedResources,
    );

    await act(async () => {
      await vi.advanceTimersByTimeAsync(SKILLS_COMMAND_DEADLINE_MS);
    });

    expect(mocks.failSkills).toHaveBeenCalledWith(
      "conversation-1",
      "skills-entry-1",
      "timeout",
    );
    expect(onSubmit).not.toHaveBeenCalled();
    vi.useRealTimers();
  });

  it("anchors deferred /help output when the command is invoked", async () => {
    let resolveSkills!: (value: {
      data: NonNullable<typeof mocks.skills>;
    }) => void;
    mocks.skills = undefined;
    mocks.refetchSkills.mockReturnValue(
      new Promise((resolve) => {
        resolveSkills = resolve;
      }),
    );
    const { submit } = setup();

    act(() => submit("/help"));
    mocks.timelineBoundaryEventId = "event-8";
    act(() =>
      resolveSkills({
        data: [
          {
            name: "review",
            type: "agentskills",
            source: "project",
            description: "Review code",
          },
        ],
      }),
    );

    await waitFor(() =>
      expect(mocks.showHelp).toHaveBeenCalledWith(
        "conversation-1",
        "event-7",
        expect.arrayContaining([
          expect.objectContaining({ command: "/review" }),
        ]),
        expect.any(Number),
      ),
    );
  });

  it("renders built-in help and warns when skill discovery fails", async () => {
    mocks.skills = undefined;
    mocks.refetchSkills.mockRejectedValue(
      new Error("Skill catalog unavailable"),
    );
    const { submit, onSubmit } = setup();

    act(() => submit("/help"));

    await waitFor(() => expect(mocks.showHelp).toHaveBeenCalledOnce());
    expect(mocks.showHelp).toHaveBeenCalledWith(
      "conversation-1",
      "event-7",
      expect.arrayContaining([expect.objectContaining({ command: "/help" })]),
      expect.any(Number),
    );
    expect(mocks.showHelp.mock.calls[0][2]).not.toEqual(
      expect.arrayContaining([expect.objectContaining({ command: "/review" })]),
    );
    expect(mocks.displayErrorToast).toHaveBeenCalledWith(
      "Skill catalog unavailable",
    );
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("renders help from the same context-aware catalog as autocomplete", () => {
    const { submit } = setup();

    act(() => submit("/help"));

    expect(mocks.showHelp).toHaveBeenCalledWith(
      "conversation-1",
      "event-7",
      expect.arrayContaining([
        expect.objectContaining({ command: "/help" }),
        expect.objectContaining({ command: "/confirm" }),
        expect.objectContaining({ command: "/fork" }),
        expect.objectContaining({ command: "/review" }),
      ]),
      expect.any(Number),
    );
  });

  it("renders context-free help on the home composer", () => {
    const { submit, onSubmit } = setupHome();

    act(() => submit("/help"));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(mocks.showHelp).toHaveBeenCalledWith(
      "home",
      null,
      expect.any(Array),
      expect.any(Number),
    );
    const commands = mocks.showHelp.mock.calls[0][2].map(
      (item: { command: string }) => item.command,
    );
    expect(commands).toEqual(
      expect.arrayContaining(["/help", "/history", "/skills", "/review"]),
    );
    expect(commands).not.toContain("/new");
    expect(commands).not.toContain("/fork");
  });

  it("renders the canonical empty /skills state on the home composer", () => {
    const { submit, onSubmit } = setupHome();

    act(() => submit("/skills"));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(mocks.refetchSkills).not.toHaveBeenCalled();
    expect(mocks.getLoadedResources).not.toHaveBeenCalled();
    expect(mocks.showSkills).toHaveBeenCalledWith(
      "home",
      null,
      {
        skills: [],
        hooks: [],
        mcps: [],
      },
      expect.any(Number),
    );
  });

  it("keeps sparse Cloud-home skill commands aligned with autocomplete", () => {
    mocks.backendKind = "cloud";
    const { submit, onSubmit } = setupHome();

    act(() => submit("/help"));

    expect(onSubmit).not.toHaveBeenCalled();
    const commands = mocks.showHelp.mock.calls[0][2].map(
      (item: { command: string }) => item.command,
    );
    const autocompleteCommands = buildSlashCommandCatalog({
      skills: mocks.skills,
      isCloud: true,
      hasConversation: false,
    }).map((item) => item.command);
    expect(commands).toEqual(autocompleteCommands);
    expect(commands).toContain("/review");
  });

  it("forks the whole local conversation and navigates to the copy", () => {
    mocks.forkConversation.mockImplementation(
      (
        _variables: unknown,
        callbacks: { onSuccess: (result: { info: { id: string } }) => void },
      ) => callbacks.onSuccess({ info: { id: "fork-2" } }),
    );
    const { submit } = setup();

    act(() => submit("/fork"));

    expect(mocks.forkConversation).toHaveBeenCalledWith(
      {
        sourceConversationId: "conversation-1",
        title: "Investigate bug (Abzweigung)",
      },
      expect.any(Object),
    );
    expect(mocks.navigate).toHaveBeenCalledWith("/conversations/fork-2");
  });
});

describe("normalizeUiCommand", () => {
  it("preserves ordinary content while trimming command formatting artifacts", () => {
    expect(normalizeUiCommand("  /he\u200Blp\uFEFF  ")).toBe("/help");
    expect(normalizeUiCommand(" ordinary prompt ")).toBe("ordinary prompt");
  });
});
