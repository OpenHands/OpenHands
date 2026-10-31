import { describe, it, expect } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EventGroup } from "#/components/conversation-events/chat/event-message-components/event-group";
import {
  ActionEvent,
  AgentErrorEvent,
  ObservationEvent,
  SecurityRisk,
  UserRejectObservation,
} from "#/types/agent-server/core";
import {
  ExecuteBashAction,
  FileEditorAction,
} from "#/types/agent-server/core/base/action";
import { ExecuteBashObservation } from "#/types/agent-server/core/base/observation";
import {
  parseSkillContent,
  styleImportantTags,
} from "#/components/conversation-events/chat/event-message-components/skill-item-expanded";
import { I18nKey } from "#/i18n/declaration";
import { SkillReadyContentList } from "#/components/conversation-events/chat/event-message-components/skill-ready-content-list";
import { SkillReadyItem } from "#/components/conversation-events/chat/event-content-helpers/create-skill-ready-event";

describe("event-group", () => {
  const makeBashAction = (
    id: string,
    command: string,
  ): ActionEvent<ExecuteBashAction> => ({
    id,
    timestamp: new Date().toISOString(),
    source: "agent",
    thought: [],
    thinking_blocks: [],
    action: {
      kind: "ExecuteBashAction",
      command,
      is_input: false,
      timeout: null,
      reset: false,
    },
    tool_name: "execute_bash",
    tool_call_id: `call_${id}`,
    tool_call: {
      id: `call_${id}`,
      type: "function",
      function: {
        name: "execute_bash",
        arguments: JSON.stringify({ command }),
      },
    },
    llm_response_id: `response_${id}`,
    security_risk: SecurityRisk.UNKNOWN,
  });

  const makeBashObservation = (
    id: string,
    actionId: string,
    command: string,
  ): ObservationEvent<ExecuteBashObservation> => ({
    id,
    timestamp: new Date().toISOString(),
    source: "environment",
    tool_name: "execute_bash",
    tool_call_id: `call_${actionId}`,
    action_id: actionId,
    observation: {
      kind: "ExecuteBashObservation",
      content: [{ type: "text", text: "ok" }],
      command,
      exit_code: 0,
      error: false,
      timeout: false,
      metadata: {} as never,
    },
  });

  const makeUserRejectObservation = (
    id: string,
    actionId: string,
  ): UserRejectObservation => ({
    id,
    timestamp: new Date().toISOString(),
    source: "environment",
    tool_name: "execute_bash",
    tool_call_id: `call_${actionId}`,
    action_id: actionId,
    rejection_reason: "User rejected the action",
  });

  const makeAgentErrorEvent = (
    id: string,
    actionId: string,
  ): AgentErrorEvent => ({
    id,
    timestamp: new Date().toISOString(),
    kind: "AgentErrorEvent",
    source: "agent",
    tool_name: "execute_bash",
    tool_call_id: `call_${actionId}`,
    error: "boom",
  });

  describe("EventGroup", () => {
    it("returns null for an empty events array", () => {
      const { container } = renderWithProviders(
        <EventGroup events={[]}>
          <div>child</div>
        </EventGroup>,
      );

      expect(container).toBeEmptyDOMElement();
    });

    it("renders a 'completed' summary when all events are observations", () => {
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashObservation("o2", "a2", "pwd"),
        makeBashObservation("o3", "a3", "whoami"),
      ];

      renderWithProviders(
        <EventGroup events={events}>
          <div data-testid="child">child content</div>
        </EventGroup>,
      );

      expect(
        screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
      ).toBeInTheDocument();
      // Children should not be visible in the collapsed state.
      expect(screen.queryByTestId("child")).not.toBeInTheDocument();
    });

    it("renders a progress summary and the running action title while in flight", () => {
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashObservation("o2", "a2", "pwd"),
        // Last action has NOT been replaced by an observation -> still running.
        makeBashAction("a3", "echo hello"),
      ];

      renderWithProviders(
        <EventGroup events={events}>
          <div>child</div>
        </EventGroup>,
      );

      expect(
        screen.getByText("EVENT_GROUP$ACTIONS_PROGRESS"),
      ).toBeInTheDocument();
      // The running action's title is rendered next to the summary. Translations
      // aren't loaded in tests, so we just verify the action's translation key
      // shows up (the localized version would interpolate the command).
      expect(screen.getByText(/ACTION_MESSAGE\$RUN/)).toBeInTheDocument();
    });

    it("keeps showing the latest completed action's title while the group is still the live tail", () => {
      // All observations -> nothing in flight, but the group has not been
      // "moved past" yet, so we expect the latest observation's title to keep
      // showing as the prominent summary alongside the completed count.
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashObservation("o2", "a2", "pwd"),
        makeBashObservation("o3", "a3", "whoami"),
      ];

      renderWithProviders(
        <EventGroup events={events}>
          <div>child</div>
        </EventGroup>,
      );

      // Latest observation's title is still in the summary line.
      expect(screen.getByText(/OBSERVATION_MESSAGE\$RUN/)).toBeInTheDocument();
      // ...next to the completed count.
      expect(
        screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
      ).toBeInTheDocument();
    });

    it("hides the latest action title once the group is finalized", () => {
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashObservation("o2", "a2", "pwd"),
        makeBashObservation("o3", "a3", "whoami"),
      ];

      renderWithProviders(
        <EventGroup events={events} isFinalized>
          <div>child</div>
        </EventGroup>,
      );

      expect(
        screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
      ).toBeInTheDocument();
      // Once moved past, we collapse to just the count — the per-action title
      // and the success check both go away.
      expect(
        screen.queryByText(/OBSERVATION_MESSAGE\$RUN/),
      ).not.toBeInTheDocument();
      expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
    });

    it("shows a spinner while running and no status icon when done", () => {
      const running = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashAction("a2", "pwd"),
      ];

      const { rerender } = renderWithProviders(
        <EventGroup events={running}>
          <div>child</div>
        </EventGroup>,
      );
      expect(screen.getByTestId("spinner-icon")).toBeInTheDocument();
      expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();

      const done = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashObservation("o2", "a2", "pwd"),
      ];
      rerender(
        <EventGroup events={done}>
          <div>child</div>
        </EventGroup>,
      );
      expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
      expect(screen.queryByTestId("status-icon")).not.toBeInTheDocument();
    });

    it("titles a settled group after its rejected action", () => {
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeUserRejectObservation("r2", "a2"),
      ];
      const allEvents = [makeBashAction("a2", "pwd"), ...events];

      renderWithProviders(
        <EventGroup events={events} allEvents={allEvents}>
          <div>child</div>
        </EventGroup>,
      );

      expect(screen.getByText(/ACTION_MESSAGE\$RUN/)).toBeInTheDocument();
      expect(
        screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
      ).toBeInTheDocument();
    });

    it("counts an action resolved by an AgentErrorEvent as resolved instead of running", () => {
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashAction("a2", "pwd"),
      ];
      const allEvents = [...events, makeAgentErrorEvent("e2", "a2")];

      renderWithProviders(
        <EventGroup events={events} allEvents={allEvents}>
          <div>child</div>
        </EventGroup>,
      );

      expect(screen.queryByTestId("spinner-icon")).not.toBeInTheDocument();
      expect(
        screen.getByText("EVENT_GROUP$ACTIONS_COMPLETED"),
      ).toBeInTheDocument();
    });

    it("updates accessibility state while toggling the group", async () => {
      const events = [
        makeBashObservation("o1", "a1", "ls"),
        makeBashObservation("o2", "a2", "pwd"),
        makeBashObservation("o3", "a3", "whoami"),
      ];
      const user = userEvent.setup();

      renderWithProviders(
        <EventGroup events={events}>
          <div data-testid="child">child content</div>
        </EventGroup>,
      );

      const toggle = screen.getByTestId("event-group-toggle");
      expect(toggle).toHaveAttribute("aria-expanded", "false");
      expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$EXPAND");
      expect(screen.queryByRole("region")).not.toBeInTheDocument();
      expect(screen.queryByTestId("child")).not.toBeInTheDocument();

      await user.click(toggle);

      const content = screen.getByRole("region");
      expect(toggle).toHaveAttribute("aria-expanded", "true");
      expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$COLLAPSE");
      expect(toggle).toHaveAttribute("aria-controls", content.id);
      expect(content).toHaveAttribute("aria-labelledby", toggle.id);
      expect(screen.getByTestId("child")).toBeInTheDocument();

      await user.click(toggle);

      expect(toggle).toHaveAttribute("aria-expanded", "false");
      expect(toggle).toHaveAttribute("aria-label", "EVENT_GROUP$EXPAND");
      expect(screen.queryByRole("region")).not.toBeInTheDocument();
      expect(screen.queryByTestId("child")).not.toBeInTheDocument();
    });

    it("keeps grouped file-action titles non-interactive inside the toggle", () => {
      const fileAction: ActionEvent<FileEditorAction> = {
        id: "a-file",
        timestamp: new Date().toISOString(),
        source: "agent",
        thought: [],
        thinking_blocks: [],
        action: {
          kind: "FileEditorAction",
          command: "str_replace",
          path: "src/app.ts",
          file_text: null,
          old_str: "a",
          new_str: "b",
          insert_line: null,
          view_range: null,
        },
        tool_name: "file_editor",
        tool_call_id: "call_a-file",
        tool_call: {
          id: "call_a-file",
          type: "function",
          function: {
            name: "file_editor",
            arguments: JSON.stringify({
              command: "str_replace",
              path: "src/app.ts",
            }),
          },
        },
        llm_response_id: "response_a-file",
        security_risk: SecurityRisk.UNKNOWN,
      };

      renderWithProviders(
        <EventGroup events={[fileAction]}>
          <div>child</div>
        </EventGroup>,
      );

      const toggle = screen.getByTestId("event-group-toggle");
      expect(
        toggle.querySelector('[data-testid="path-component-link"]'),
      ).toBeNull();
      expect(toggle.querySelectorAll("button")).toHaveLength(0);
    });
  });
});

describe("skill-item-expanded", () => {
  describe("parseSkillContent", () => {
    it("should extract matchInfo from content", () => {
      const content =
        "The following information has been included based on keyword match\nIt may or may not be relevant\n\nActual body content";
      const result = parseSkillContent(content);

      expect(result.matchInfo).toBe(
        "The following information has been included based on keyword match",
      );
      expect(result.body).toBe("Actual body content");
    });

    it("should extract filePath from content", () => {
      const content = "Skill location: /path/to/skill.md\n\nBody here";
      const result = parseSkillContent(content);

      expect(result.filePath).toBe("/path/to/skill.md");
      expect(result.body).toBe("Body here");
    });

    it("should handle content with both matchInfo and filePath", () => {
      const content =
        "The following information has been included based on keyword match\nIt may or may not be relevant\nSkill location: /path/to/skill.md\n(Use this path to resolve imports)\n\nBody content";
      const result = parseSkillContent(content);

      expect(result.matchInfo).toBe(
        "The following information has been included based on keyword match",
      );
      expect(result.filePath).toBe("/path/to/skill.md");
      expect(result.body).toBe("Body content");
    });

    it("should handle content with no metadata", () => {
      const content = "Just plain body content\nWith multiple lines";
      const result = parseSkillContent(content);

      expect(result.matchInfo).toBeNull();
      expect(result.filePath).toBeNull();
      expect(result.body).toBe("Just plain body content\nWith multiple lines");
    });

    it("should handle empty content", () => {
      const result = parseSkillContent("");

      expect(result.matchInfo).toBeNull();
      expect(result.filePath).toBeNull();
      expect(result.body).toBe("");
    });

    it("should skip blank lines between metadata and body", () => {
      const content =
        "Skill location: /path/to/skill.md\n\n\nBody after blanks";
      const result = parseSkillContent(content);

      expect(result.filePath).toBe("/path/to/skill.md");
      expect(result.body).toBe("Body after blanks");
    });
  });

  describe("styleImportantTags", () => {
    it("should wrap important tag content with bold markers", () => {
      const text = "Some text <important>critical info</important> more text";
      const result = styleImportantTags(text);

      expect(result).toBe("Some text **critical info** more text");
    });

    it("should handle multiple important tags", () => {
      const text =
        "<important>first</important> and <important>second</important>";
      const result = styleImportantTags(text);

      expect(result).toBe("**first** and **second**");
    });

    it("should handle multiline content inside important tags", () => {
      const text = "<important>line1\nline2</important>";
      const result = styleImportantTags(text);

      expect(result).toBe("**line1\nline2**");
    });

    it("should be case-insensitive", () => {
      const text = "<IMPORTANT>test</IMPORTANT>";
      const result = styleImportantTags(text);

      expect(result).toBe("**test**");
    });

    it("should return text unchanged if no important tags", () => {
      const text = "No important tags here";
      const result = styleImportantTags(text);

      expect(result).toBe("No important tags here");
    });

    it("should trim whitespace inside important tags", () => {
      const text = "<important>  spaced  </important>";
      const result = styleImportantTags(text);

      expect(result).toBe("**spaced**");
    });
  });
});

describe("skill-ready-content-list", () => {
  const makeItems = (...entries: [string, string][]): SkillReadyItem[] =>
    entries.map(([name, content]) => ({ name, content }));

  describe("SkillReadyContentList", () => {
    it("renders all skill names", () => {
      const items = makeItems(["docker", "content1"], ["gitlab", "content2"]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      expect(screen.getByText("docker")).toBeInTheDocument();
      expect(screen.getByText("gitlab")).toBeInTheDocument();
    });

    it("renders the header label", () => {
      const items = makeItems(["docker", "content"]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      expect(
        screen.getByText("SKILLS$TRIGGERED_SKILL_KNOWLEDGE"),
      ).toBeInTheDocument();
    });

    it("renders a custom header label when titleKey is provided", () => {
      const items = makeItems(["docker", "content"]);

      renderWithProviders(
        <SkillReadyContentList
          items={items}
          titleKey={I18nKey.SKILLS$INVOKED_SKILL_KNOWLEDGE}
        />,
      );

      expect(
        screen.getByText("SKILLS$INVOKED_SKILL_KNOWLEDGE"),
      ).toBeInTheDocument();
      expect(
        screen.queryByText("SKILLS$TRIGGERED_SKILL_KNOWLEDGE"),
      ).not.toBeInTheDocument();
    });

    it("does not show content before clicking", () => {
      const items = makeItems(["docker", "Docker usage guide"]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      expect(screen.queryByText("Docker usage guide")).not.toBeInTheDocument();
    });

    it("expands skill content on click", async () => {
      const user = userEvent.setup();
      const items = makeItems(["docker", "Docker usage guide"]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      await user.click(screen.getByText("docker"));

      expect(screen.getByText("Docker usage guide")).toBeInTheDocument();
    });

    it("collapses skill content on second click", async () => {
      const user = userEvent.setup();
      const items = makeItems(["docker", "Docker usage guide"]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      await user.click(screen.getByText("docker"));
      expect(screen.getByText("Docker usage guide")).toBeInTheDocument();

      await user.click(screen.getByText("docker"));
      expect(screen.queryByText("Docker usage guide")).not.toBeInTheDocument();
    });

    it("expands skills independently", async () => {
      const user = userEvent.setup();
      const items = makeItems(
        ["docker", "Docker guide"],
        ["gitlab", "GitLab guide"],
      );

      renderWithProviders(<SkillReadyContentList items={items} />);

      await user.click(screen.getByText("docker"));

      expect(screen.getByText("Docker guide")).toBeInTheDocument();
      expect(screen.queryByText("GitLab guide")).not.toBeInTheDocument();
    });

    it("renders <important> content as bold text", async () => {
      const user = userEvent.setup();
      const content =
        "Some text <important>critical info</important> more text";
      const items = makeItems(["docker", content]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      await user.click(screen.getByText("docker"));

      // The important text should be rendered as bold (strong element)
      const boldElement = screen.getByText("critical info");
      expect(boldElement).toBeInTheDocument();
      expect(boldElement.tagName).toBe("STRONG");
    });

    it("parses and displays file path from metadata", async () => {
      const user = userEvent.setup();
      const content = [
        'The following information has been included based on a keyword match for "docker".',
        "It may or may not be relevant to the user's request.",
        "Skill location: /home/openhands/.openhands/skills/docker/SKILL.md",
        "(Use this path to resolve relative file references)",
        "",
        "Docker Usage Guide",
      ].join("\n");
      const items = makeItems(["docker", content]);

      renderWithProviders(<SkillReadyContentList items={items} />);

      await user.click(screen.getByText("docker"));

      // File path rendered in code element
      expect(
        screen.getByText("/home/openhands/.openhands/skills/docker/SKILL.md"),
      ).toBeInTheDocument();
      // Actual skill body rendered
      expect(screen.getByText("Docker Usage Guide")).toBeInTheDocument();
      // Metadata preamble lines are not rendered as-is in the body
      expect(
        screen.queryByText("It may or may not be relevant"),
      ).not.toBeInTheDocument();
    });
  });
});
