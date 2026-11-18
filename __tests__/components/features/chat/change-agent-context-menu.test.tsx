import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { compile } from "@tailwindcss/node";
import { parse } from "postcss";
import { fireEvent, screen, render } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { renderWithProviders } from "test-utils";
import { ChangeAgentContextMenu } from "#/components/features/chat/change-agent-context-menu";
import i18n from "i18next";
import userEvent from "@testing-library/user-event";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { useModelStore } from "#/stores/model-store";
import { useFreeModelsStore } from "#/stores/free-models-store";
import { type ProfileInfo } from "#/api/profiles-service/profiles-service.api";
import { WorkspaceModeSelector } from "#/components/features/chat/workspace-mode-selector";

describe("ChangeAgentContextMenu", () => {
  it("marks the active mode and preserves its compact description type", async () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="code"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(codeOption).toHaveClass("bg-interactive-hover");
    expect(planOption).not.toHaveClass("bg-interactive-hover");
    const description = codeOption.querySelector(".whitespace-pre-wrap");
    if (!description) throw new Error("Expected the mode description");
    const compiler = await compile(
      await readFile(resolve("src/tailwind.css"), "utf8"),
      { base: resolve("src"), onDependency: () => {} },
    );
    const rules: string[] = [];
    parse(compiler.build([...description.classList])).walkDecls(
      "font-size",
      (declaration) => {
        const rule = declaration.parent;
        if (rule?.type === "rule" && description.matches(rule.selector)) {
          rules.push(`${rule.selector}{${declaration.toString()}}`);
        }
      },
    );
    const style = document.createElement("style");
    style.textContent = rules.join("\n");
    document.head.appendChild(style);
    try {
      expect(getComputedStyle(description).fontSize).toBe("10px");
    } finally {
      style.remove();
    }

    expect(codeOption.querySelectorAll("svg")).toHaveLength(2);
    expect(planOption.querySelectorAll("svg")).toHaveLength(1);
  });

  it("shows plan as active when conversation mode is plan", () => {
    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={vi.fn()}
        onCodeClick={vi.fn()}
        onPlanClick={vi.fn()}
      />,
    );

    const codeOption = screen.getByTestId("code-option");
    const planOption = screen.getByTestId("plan-option");

    expect(planOption).toHaveClass("bg-interactive-hover");
    expect(codeOption).not.toHaveClass("bg-interactive-hover");
    expect(planOption.querySelectorAll("svg")).toHaveLength(2);
  });

  it("calls onCodeClick and onClose when code is selected", () => {
    const onClose = vi.fn();
    const onCodeClick = vi.fn();

    renderWithProviders(
      <ChangeAgentContextMenu
        activeMode="plan"
        onClose={onClose}
        onCodeClick={onCodeClick}
        onPlanClick={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByTestId("code-option"));

    expect(onCodeClick).toHaveBeenCalledTimes(1);
    expect(onClose).toHaveBeenCalledTimes(1);
  });
});

describe("model-messages", () => {
  const CONVERSATION_ID = "conv-1";

  const profiles: ProfileInfo[] = [
    {
      name: "haiku",
      model: "anthropic/claude-haiku-4-5",
      base_url: "https://llm.example.test",
      api_key_set: true,
    },
    {
      name: "gpt",
      model: "openai/gpt-5.1",
      base_url: null,
      api_key_set: false,
    },
  ];

  describe("ModelMessages", () => {
    beforeEach(() => {
      const resources = {
        MODEL$AVAILABLE_PROFILES: "Available profiles ({{count}})",
        MODEL$NO_SAVED_PROFILES: "No saved profiles",
        MODEL$NO_PROFILES_HINT:
          "Use the LLM settings page to create a profile, then run /model <name> to switch.",
        MODEL$USER_SWITCHED: "You switched models to <cmd>{{name}}</cmd>",
        MODEL$AGENT_SWITCHED: "Agent switched models to <cmd>{{name}}</cmd>",
        MODEL$ROUTER_SWITCHED:
          "Model router switched models to <cmd>{{name}}</cmd>",
        MODEL$AGENT_SWITCH_REASON: "Agent’s reason:",
        MODEL$ROUTER_SWITCH_REASON: "Model router’s reason:",
        MODEL$SWITCHED_TO_PROFILE: "ℹ️ Switched to profile <cmd>{{name}}</cmd>",
      };
      i18n.addResourceBundle("en", "translation", resources, true, true);
      i18n.addResourceBundle("en", "openhands", resources, true, true);
      useModelStore.setState({ entriesByConversation: {} });
      useFreeModelsStore.getState().setFlags({
        freeModels: new Set(),
        defaultModel: null,
      });
    });

    it("renders only entries anchored to the requested event", async () => {
      const user = userEvent.setup();
      useModelStore.getState().show(CONVERSATION_ID, "event-1", [profiles[0]]);
      useModelStore.getState().show(CONVERSATION_ID, "event-2", [profiles[1]]);

      renderWithProviders(
        <ModelMessages
          conversationId={CONVERSATION_ID}
          anchorEventId="event-1"
        />,
      );

      expect(screen.getByTestId("model-messages")).toBeInTheDocument();
      expect(screen.getByText("Available profiles (1)")).toBeInTheDocument();

      await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));

      const profileToggle = screen.getByRole("button", {
        name: "Toggle details for haiku",
      });
      expect(profileToggle).toHaveAttribute("aria-expanded", "false");
      expect(
        screen.queryByRole("button", { name: "Toggle details for gpt" }),
      ).not.toBeInTheDocument();

      await user.click(profileToggle);
      expect(profileToggle).toHaveAttribute("aria-expanded", "true");

      expect(
        screen.getByText(/model:\s+anthropic\/claude-haiku-4-5/),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/base_url:\s+https:\/\/llm\.example\.test/),
      ).toBeInTheDocument();
      expect(screen.getByText(/api_key:\s+set/)).toBeInTheDocument();
    });

    it("renders empty-profile hints expanded by default", () => {
      useModelStore.getState().show(CONVERSATION_ID, null, []);

      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );

      expect(screen.getByText("No saved profiles")).toBeInTheDocument();
      expect(
        screen.getByText(
          "Use the LLM settings page to create a profile, then run /model <name> to switch.",
        ),
      ).toBeInTheDocument();
    });

    it("renders switch confirmations for the matching anchor", () => {
      useModelStore
        .getState()
        .recordSwitch(CONVERSATION_ID, "event-1", "haiku");

      renderWithProviders(
        <ModelMessages
          conversationId={CONVERSATION_ID}
          anchorEventId="event-1"
        />,
      );

      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "You switched models to haiku",
      );
    });

    it("shows an agent's rationale immediately as plain text", () => {
      useModelStore.getState().recordSwitch(CONVERSATION_ID, null, "astra", {
        initiator: "agent",
        reason: "Careful audit <script>alert(1)</script>",
      });
      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );
      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "Agent switched models to astra",
      );
      expect(screen.getByText("Agent’s reason:")).toBeVisible();
      expect(
        screen.getByText("Careful audit <script>alert(1)</script>"),
      ).toBeVisible();
      expect(document.querySelector("script")).toBeNull();
    });

    it("distinguishes router switches and omits unavailable reasons", () => {
      useModelStore.getState().recordSwitch(CONVERSATION_ID, null, "fast", {
        initiator: "router",
      });
      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );
      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "Model router switched models to fast",
      );
      expect(screen.queryByText("Agent’s reason:")).not.toBeInTheDocument();
      expect(
        screen.queryByText("Model router’s reason:"),
      ).not.toBeInTheDocument();
    });

    it("shows the router's chosen class as its reason when present", () => {
      useModelStore.getState().recordSwitch(CONVERSATION_ID, null, "fast", {
        initiator: "router",
        reason: "model: fast",
      });
      renderWithProviders(
        <ModelMessages conversationId={CONVERSATION_ID} anchorEventId={null} />,
      );
      expect(screen.getByTestId("model-messages")).toHaveTextContent(
        "Model router switched models to fast",
      );
      expect(screen.getByText("Model router’s reason:")).toBeVisible();
      expect(screen.getByText("model: fast")).toBeVisible();
      expect(screen.queryByText("Agent’s reason:")).not.toBeInTheDocument();
    });

    it("labels a DB-flagged free OpenHands route in expanded profile diagnostics", async () => {
      const user = userEvent.setup();
      useFreeModelsStore.getState().setFlags({
        freeModels: new Set(["openhands/glm-5.2"]),
        defaultModel: "openhands/glm-5.2",
      });
      useModelStore.getState().show(CONVERSATION_ID, "event-1", [
        {
          name: "free",
          model: "openhands/glm-5.2",
          base_url: null,
          api_key_set: true,
        },
      ]);

      renderWithProviders(
        <ModelMessages
          conversationId={CONVERSATION_ID}
          anchorEventId="event-1"
        />,
      );

      await user.click(screen.getByRole("button", { name: "BUTTON$EXPAND" }));
      await user.click(
        screen.getByRole("button", { name: "Toggle details for free" }),
      );

      expect(
        screen.getByText(/model:\s+openhands\/glm-5\.2 \(free\)/),
      ).toBeInTheDocument();
    });
  });
});

describe("WorkspaceModeSelector", () => {
  it("opens the menu and calls onChange when an option is selected", () => {
    const onChange = vi.fn();
    render(
      <WorkspaceModeSelector
        value="local_repo"
        options={["local_repo", "new_worktree"]}
        backendKind="local"
        onChange={onChange}
      />,
    );

    // Open the menu (trigger renders the active mode via WorkspaceModeIcon).
    fireEvent.click(screen.getByTestId("workspace-mode-selector"));
    fireEvent.click(
      screen.getByTestId("workspace-mode-selector-option-new_worktree"),
    );

    expect(onChange).toHaveBeenCalledWith("new_worktree");
  });
});
