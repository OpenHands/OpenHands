import { act, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { SlashCommandMessages } from "#/components/features/chat/slash-command-messages";
import { useSlashCommandOutputStore } from "#/stores/slash-command-output-store";

describe("SlashCommandMessages", () => {
  afterEach(() => {
    act(() => useSlashCommandOutputStore.getState().clearAll());
  });

  it("renders loading immediately and completes the same DOM entry in place", () => {
    let entryId = "";
    act(() => {
      entryId = useSlashCommandOutputStore
        .getState()
        .beginSkills("conversation-1", null);
    });

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId={null}
      />,
    );

    const entry = screen.getByTestId(`slash-command-skills-${entryId}`);
    expect(entry).toHaveAttribute("data-status", "loading");
    expect(screen.getByRole("status")).toBeVisible();
    expect(screen.getByTestId("loading-spinner")).toBeVisible();
    expect(screen.getByText("SLASH_COMMAND$LOADING_RESOURCES")).toBeVisible();

    act(() =>
      useSlashCommandOutputStore
        .getState()
        .completeSkills("conversation-1", entryId, {
          skills: [],
          hooks: [],
          mcps: [],
        }),
    );

    expect(
      screen.getByTestId(`slash-command-skills-${entryId}`),
    ).toHaveAttribute("data-status", "ready");
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.getAllByTestId("slash-command-messages")).toHaveLength(1);
    expect(screen.getByTestId("slash-command-skills-list")).toBeVisible();
  });

  it("renders fallback help immediately and enriches the same DOM entry", () => {
    const fallback = [
      {
        command: "/help",
        skill: { name: "help", type: "agentskills" as const, source: null },
      },
    ];
    let entryId = "";
    act(() => {
      entryId = useSlashCommandOutputStore
        .getState()
        .beginHelp("conversation-1", null, fallback);
    });

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId={null}
      />,
    );

    const entry = screen.getByTestId(`slash-command-help-${entryId}`);
    expect(entry).toHaveAttribute("data-status", "loading");
    expect(screen.getByText("/help")).toBeVisible();
    expect(screen.getByTestId("slash-command-help-loading")).toBeVisible();

    act(() =>
      useSlashCommandOutputStore
        .getState()
        .completeHelp("conversation-1", entryId, [
          ...fallback,
          {
            command: "/review",
            skill: {
              name: "review",
              type: "agentskills",
              source: "project",
              description: "Review code",
            },
          },
        ]),
    );

    expect(screen.getByTestId(`slash-command-help-${entryId}`)).toBe(entry);
    expect(entry).toHaveAttribute("data-status", "ready");
    expect(
      screen.queryByTestId("slash-command-help-loading"),
    ).not.toBeInTheDocument();
    fireEvent.click(screen.getByText("SLASH_COMMAND$SKILL_COMMANDS"));
    expect(screen.getByText("/review")).toBeVisible();
  });

  it.each([
    ["request", "SLASH_COMMAND$RESOURCES_FAILED", false],
    ["timeout", "SLASH_COMMAND$RESOURCES_TIMEOUT", true],
  ] as const)(
    "renders a persistent %s error with retry guidance",
    (errorKind, title, hasTimeoutIcon) => {
      let entryId = "";
      act(() => {
        const store = useSlashCommandOutputStore.getState();
        entryId = store.beginSkills("conversation-1", null);
        store.failSkills("conversation-1", entryId, errorKind);
      });

      render(
        <SlashCommandMessages
          outputScopeId="conversation-1"
          timelineBoundaryEventId={null}
        />,
      );

      expect(
        screen.getByTestId(`slash-command-skills-${entryId}`),
      ).toHaveAttribute("data-status", "error");
      expect(screen.getByText(title)).toBeVisible();
      expect(
        screen.getByTestId("slash-command-skills-error"),
      ).toHaveTextContent("SLASH_COMMAND$SKILLS_RETRY");
      expect(screen.queryByTestId("status-icon") !== null).toBe(hasTimeoutIcon);
    },
  );

  it("renders only skill results anchored at the requested event", () => {
    act(() => {
      const { showSkills } = useSlashCommandOutputStore.getState();
      showSkills("conversation-1", "event-1", {
        skills: [
          {
            name: "review",
            source: "project",
            description: "Review the current changes",
          },
        ],
        hooks: [{ hookType: "pre_tool_use", commands: ["lint", "test"] }],
        mcps: [{ name: "github", transport: "stdio" }],
      });
      showSkills("conversation-1", "event-2", {
        skills: [
          {
            name: "release",
            source: "public",
            description: "Prepare release notes",
          },
        ],
        hooks: [],
        mcps: [],
      });
    });

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId="event-1"
      />,
    );

    expect(screen.getByText("SLASH_COMMAND$LOADED_RESOURCES")).toBeVisible();
    expect(screen.getByText("SLASH_COMMAND$SUMMARY:")).toBeVisible();
    expect(screen.getByText("• review")).toBeVisible();
    expect(screen.getByText("Review the current changes")).toBeVisible();
    expect(screen.getByText("(project)")).toBeVisible();
    expect(screen.getByText("• pre_tool_use: lint, test")).toBeVisible();
    expect(screen.getByText("• github")).toBeVisible();
    expect(screen.getByText("(stdio)")).toBeVisible();
    expect(screen.queryByText("release")).not.toBeInTheDocument();
    expect(screen.getByTestId("slash-command-skills-list")).toHaveClass(
      "custom-scrollbar-always",
      "max-h-[50vh]",
      "overflow-y-auto",
    );
  });

  it("renders an explicit empty result", () => {
    act(() =>
      useSlashCommandOutputStore.getState().showSkills("conversation-1", null, {
        skills: [],
        hooks: [],
        mcps: [],
      }),
    );

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId={null}
      />,
    );

    expect(screen.getByText("SLASH_COMMAND$LOADED_RESOURCES")).toBeVisible();
    expect(
      screen.getByText("SLASH_COMMAND$NO_RESOURCES_SUMMARY"),
    ).toBeVisible();
    expect(screen.getByText("SLASH_COMMAND$NO_LOADED_RESOURCES")).toBeVisible();
  });

  it("renders built-ins and exposes skill descriptions as tooltips", () => {
    act(() =>
      useSlashCommandOutputStore
        .getState()
        .showHelp("conversation-1", "event-1", [
          {
            command: "/new",
            skill: {
              name: "new",
              type: "agentskills",
              source: null,
              content: "Start a new conversation",
            },
          },
          {
            command: "/review",
            skill: {
              name: "review",
              type: "agentskills",
              source: "project",
              description: "Review the current changes",
            },
          },
          {
            command: "/fork",
            skill: {
              name: "fork",
              type: "agentskills",
              source: null,
              content: "Copy this conversation into a new conversation",
            },
          },
          {
            command: "/model",
            skill: {
              name: "model",
              type: "agentskills",
              source: null,
              content: "Switch the active model",
            },
          },
        ]),
    );

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId="event-1"
      />,
    );

    expect(screen.getByText("SLASH_COMMAND$AVAILABLE_COMMANDS")).toBeVisible();
    expect(screen.getByText("SLASH_COMMAND$CLI_COMMANDS")).toBeVisible();
    expect(screen.getByText("SLASH_COMMAND$CANVAS_COMMANDS")).toBeVisible();
    const skillCommands = screen.getByTestId(
      "slash-command-help-skill-commands",
    );
    expect(skillCommands).not.toHaveAttribute("open");
    expect(screen.getByText("SLASH_COMMAND$SKILL_COMMANDS")).toBeVisible();
    expect(screen.getByText("/new")).toBeVisible();
    expect(screen.getByText("Start a new conversation")).toBeVisible();
    expect(screen.getByText("/fork")).toBeVisible();
    expect(screen.getByText("/model")).toBeVisible();
    const reviewCommand = screen.getByText("/review");
    expect(reviewCommand).not.toBeVisible();
    fireEvent.click(screen.getByText("SLASH_COMMAND$SKILL_COMMANDS"));
    expect(skillCommands).toHaveAttribute("open");
    expect(reviewCommand).toBeVisible();
    expect(reviewCommand).toHaveAttribute("tabindex", "0");
    expect(reviewCommand).toHaveAccessibleName(
      "/review: Review the current changes",
    );
    expect(
      screen.queryByText("Review the current changes"),
    ).not.toBeInTheDocument();
    expect(screen.getByText("SLASH_COMMAND$AUTOCOMPLETE_TIP")).toBeVisible();
    const helpList = screen.getByTestId("slash-command-help-list");
    expect(helpList).toHaveClass(
      "custom-scrollbar-always",
      "max-h-[50vh]",
      "overflow-y-auto",
    );
    expect(
      within(helpList)
        .getAllByText(/^\/.+/)
        .map((element) => element.textContent),
    ).toEqual(["/new", "/fork", "/model", "/review"]);
  });

  it("distinguishes unavailable Cloud resource categories from empty ones", () => {
    act(() =>
      useSlashCommandOutputStore.getState().showSkills("conversation-1", null, {
        skills: [],
        hooks: null,
        mcps: null,
      }),
    );

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId={null}
      />,
    );

    const resourceList = screen.getByTestId("slash-command-skills-list");
    expect(resourceList).toHaveTextContent("SLASH_COMMAND$SKILL_COUNT_other");
    expect(resourceList).toHaveTextContent("SLASH_COMMAND$HOOKS_UNAVAILABLE");
    expect(resourceList).toHaveTextContent("SLASH_COMMAND$MCPS_UNAVAILABLE");
    expect(
      screen.queryByText("SLASH_COMMAND$NO_LOADED_RESOURCES"),
    ).not.toBeInTheDocument();
  });

  it("omits a successfully empty Hooks category without marking it unavailable", () => {
    act(() =>
      useSlashCommandOutputStore.getState().showSkills("conversation-1", null, {
        skills: [{ name: "review", description: "Review code", source: null }],
        hooks: [],
        mcps: null,
      }),
    );

    render(
      <SlashCommandMessages
        outputScopeId="conversation-1"
        timelineBoundaryEventId={null}
      />,
    );

    const resourceList = screen.getByTestId("slash-command-skills-list");
    expect(resourceList).not.toHaveTextContent("SLASH_COMMAND$HOOKS_SECTION");
    expect(resourceList).not.toHaveTextContent(
      "SLASH_COMMAND$HOOKS_UNAVAILABLE",
    );
  });
});
