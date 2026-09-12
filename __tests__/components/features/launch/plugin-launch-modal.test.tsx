import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { PluginLaunchModal } from "#/components/features/launch/plugin-launch-modal";
import { PluginSpec } from "#/api/conversation-service/agent-server-conversation-service.types";

const mockOnStartConversation = vi.fn();
const mockOnClose = vi.fn();

function renderModal(
  plugins: PluginSpec[],
  props: Partial<{
    message: string;
    isLoading: boolean;
  }> = {},
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <PluginLaunchModal
        plugins={plugins}
        message={props.message}
        isLoading={props.isLoading ?? false}
        onStartConversation={mockOnStartConversation}
        onClose={mockOnClose}
      />
    </QueryClientProvider>,
  );
}

describe("PluginLaunchModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe("Plugin Display Name Extraction", () => {
    it("should extract plugin name from repo_path when provided", () => {
      renderModal([
        { source: "github:owner/repo", repo_path: "plugins/my-plugin" },
      ]);

      // Plugin name should be "my-plugin" from the path
      expect(screen.getByText("my-plugin")).toBeInTheDocument();
    });

    it("should show repo path when no repo_path (repo IS the plugin)", () => {
      renderModal([{ source: "github:owner/my-plugin" }]);

      // When no repo_path, the whole repo is the plugin, show "owner/my-plugin"
      const elements = screen.getAllByText("owner/my-plugin");
      expect(elements.length).toBeGreaterThan(0);
    });

    it("should extract name from git URL", () => {
      renderModal([{ source: "https://github.com/owner/repo-name.git" }]);

      const elements = screen.getAllByText("repo-name");
      expect(elements.length).toBeGreaterThan(0);
    });

    it("should display full source when no special format", () => {
      renderModal([{ source: "local-plugin" }]);

      const elements = screen.getAllByText("local-plugin");
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  describe("Modal Title", () => {
    it("should show plugin name in title for single plugin", () => {
      renderModal([{ source: "github:owner/awesome-plugin" }]);

      // Title should include the plugin name - use getAllBy since text appears in multiple places
      const elements = screen.getAllByText(/owner\/awesome-plugin/);
      expect(elements.length).toBeGreaterThan(0);
    });

    it("should show generic title for multiple plugins", () => {
      renderModal([
        { source: "github:owner/plugin1" },
        { source: "github:owner/plugin2" },
      ]);

      // The h2 title contains both LAUNCH$MODAL_TITLE and LAUNCH$MODAL_TITLE_GENERIC
      const title = screen.getByRole("heading", { level: 2 });
      expect(title.textContent).toContain("LAUNCH$MODAL_TITLE_GENERIC");
    });
  });

  describe("Message Display", () => {
    it("should display message when provided", () => {
      renderModal([{ source: "github:owner/repo" }], {
        message: "This is a custom message",
      });

      expect(screen.getByText("This is a custom message")).toBeInTheDocument();
    });

    it("should not render message element when not provided", () => {
      renderModal([{ source: "github:owner/repo" }]);

      // No message should be present
      const modal = screen.getByTestId("plugin-launch-modal");
      expect(modal.querySelector("p.text-neutral-400")).not.toBeInTheDocument();
    });
  });

  describe("Expandable Sections", () => {
    it("should expand plugin section by default when it has parameters", () => {
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { apiKey: "test-key" },
        },
      ]);

      // Parameter input should be visible (section is expanded)
      expect(screen.getByTestId("plugin-0-param-apiKey")).toBeInTheDocument();
    });

    it("should collapse/expand section when clicking header", async () => {
      const user = userEvent.setup();
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { apiKey: "test-key" },
        },
      ]);

      // Initially expanded - parameter visible
      expect(screen.getByTestId("plugin-0-param-apiKey")).toBeInTheDocument();

      // Click to collapse
      await user.click(screen.getByTestId("plugin-section-0"));

      // Parameter should be hidden
      await waitFor(() => {
        expect(
          screen.queryByTestId("plugin-0-param-apiKey"),
        ).not.toBeInTheDocument();
      });

      // Click to expand again
      await user.click(screen.getByTestId("plugin-section-0"));

      // Parameter should be visible again
      await waitFor(() => {
        expect(screen.getByTestId("plugin-0-param-apiKey")).toBeInTheDocument();
      });
    });
  });

  describe("Parameter Inputs", () => {
    it("should render text input for string parameters", () => {
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { name: "default-name" },
        },
      ]);

      const input = screen.getByTestId("plugin-0-param-name");
      expect(input).toHaveAttribute("type", "text");
      expect(input).toHaveValue("default-name");
    });

    it("should render number input for number parameters", () => {
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { count: 42 },
        },
      ]);

      const input = screen.getByTestId("plugin-0-param-count");
      expect(input).toHaveAttribute("type", "number");
      expect(input).toHaveValue(42);
    });

    it("should render checkbox for boolean parameters", () => {
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { enabled: true },
        },
      ]);

      const checkbox = screen.getByTestId("plugin-0-param-enabled");
      expect(checkbox).toHaveAttribute("type", "checkbox");
      expect(checkbox).toBeChecked();
    });

    it("should update string parameter value when typing", async () => {
      const user = userEvent.setup();
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { apiKey: "initial" },
        },
      ]);

      const input = screen.getByTestId("plugin-0-param-apiKey");
      await user.clear(input);
      await user.type(input, "new-value");

      expect(input).toHaveValue("new-value");
    });

    it("should update number parameter value when typing", async () => {
      const user = userEvent.setup();
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { count: 5 },
        },
      ]);

      const input = screen.getByTestId("plugin-0-param-count");
      await user.clear(input);
      await user.type(input, "100");

      expect(input).toHaveValue(100);
    });

    it("should toggle boolean parameter when clicking checkbox", async () => {
      const user = userEvent.setup();
      renderModal([
        {
          source: "github:owner/repo",
          parameters: { debug: false },
        },
      ]);

      const checkbox = screen.getByTestId("plugin-0-param-debug");
      expect(checkbox).not.toBeChecked();

      await user.click(checkbox);

      expect(checkbox).toBeChecked();
    });
  });

  describe("Ref and Path Display", () => {
    it("should display ref when provided", async () => {
      renderModal([
        {
          source: "github:owner/repo",
          ref: "v1.2.3",
          parameters: { key: "value" },
        },
      ]);

      // Check the expanded section contains the ref value
      const modal = screen.getByTestId("plugin-launch-modal");
      expect(modal.textContent).toContain("v1.2.3");
    });

    it("should display repo_path when provided", () => {
      renderModal([
        {
          source: "github:owner/repo",
          repo_path: "plugins/my-plugin",
          parameters: { key: "value" },
        },
      ]);

      // Check the expanded section contains the path value
      const modal = screen.getByTestId("plugin-launch-modal");
      expect(modal.textContent).toContain("plugins/my-plugin");
    });
  });

  describe("Plugins Without Parameters", () => {
    it("should show plugins list when all plugins have no parameters", () => {
      renderModal([
        { source: "github:owner/plugin1" },
        { source: "github:owner/plugin2" },
      ]);

      expect(screen.getByText("LAUNCH$PLUGINS")).toBeInTheDocument();
      // When no repo_path, the full repo path is shown (may appear multiple times)
      expect(screen.getAllByText("owner/plugin1").length).toBeGreaterThan(0);
      expect(screen.getAllByText("owner/plugin2").length).toBeGreaterThan(0);
    });

    it("should show 'Additional Plugins' when mixing plugins with and without params", () => {
      renderModal([
        { source: "github:owner/with-params", parameters: { key: "val" } },
        { source: "github:owner/without-params" },
      ]);

      expect(screen.getByText("LAUNCH$ADDITIONAL_PLUGINS")).toBeInTheDocument();
      // When no repo_path, the full repo path is shown
      expect(
        screen.getAllByText("owner/without-params").length,
      ).toBeGreaterThan(0);
    });

    it("should show ref in simple plugin list", () => {
      renderModal([{ source: "github:owner/plugin", ref: "main" }]);

      expect(screen.getByText("@ main")).toBeInTheDocument();
    });

    it("should show repo_path in simple plugin list", () => {
      renderModal([
        { source: "github:owner/repo", repo_path: "plugins/city-weather" },
      ]);

      // Should show the plugin name
      expect(screen.getByText("city-weather")).toBeInTheDocument();
      // Should show the source info with path
      const modal = screen.getByTestId("plugin-launch-modal");
      expect(modal.textContent).toContain("owner/repo");
      expect(modal.textContent).toContain("plugins/city-weather");
    });
  });

  describe("Action Buttons", () => {
    it("should call onClose when close button is clicked", async () => {
      const user = userEvent.setup();
      renderModal([{ source: "github:owner/repo" }]);

      await user.click(screen.getByTestId("close-button"));

      expect(mockOnClose).toHaveBeenCalledTimes(1);
    });

    it("should call onStartConversation with updated plugins when start button is clicked", async () => {
      const user = userEvent.setup();
      renderModal([
        {
          source: "github:owner/repo",
          ref: "main",
          parameters: { apiKey: "initial" },
        },
      ]);

      // Update the parameter
      const input = screen.getByTestId("plugin-0-param-apiKey");
      await user.clear(input);
      await user.type(input, "updated-key");

      // Check the trust checkbox first
      await user.click(screen.getByTestId("trust-checkbox"));

      // Click start
      await user.click(screen.getByTestId("start-conversation-button"));

      expect(mockOnStartConversation).toHaveBeenCalledTimes(1);
      const calledWithPlugins = mockOnStartConversation.mock.calls[0][0];
      const calledWithMessage = mockOnStartConversation.mock.calls[0][1];
      expect(calledWithPlugins[0].source).toBe("github:owner/repo");
      expect(calledWithPlugins[0].ref).toBe("main");
      expect(calledWithPlugins[0].parameters.apiKey).toBe("updated-key");
      expect(calledWithMessage).toBeUndefined();
    });

    it("should call onStartConversation with message when provided", async () => {
      const user = userEvent.setup();
      renderModal([{ source: "github:owner/repo" }], {
        message: "/city-weather:now Tokyo",
      });

      // Check the trust checkbox first
      await user.click(screen.getByTestId("trust-checkbox"));

      await user.click(screen.getByTestId("start-conversation-button"));

      expect(mockOnStartConversation).toHaveBeenCalledTimes(1);
      const calledWithPlugins = mockOnStartConversation.mock.calls[0][0];
      const calledWithMessage = mockOnStartConversation.mock.calls[0][1];
      expect(calledWithPlugins[0].source).toBe("github:owner/repo");
      expect(calledWithMessage).toBe("/city-weather:now Tokyo");
    });

    it("should show 'Starting...' text when loading", () => {
      renderModal([{ source: "github:owner/repo" }], { isLoading: true });

      expect(screen.getByText("LAUNCH$STARTING")).toBeInTheDocument();
    });

    it("should disable start button when loading", () => {
      renderModal([{ source: "github:owner/repo" }], { isLoading: true });

      expect(screen.getByTestId("start-conversation-button")).toBeDisabled();
    });
  });

  describe("Multiple Plugins with Parameters", () => {
    it("should render multiple expandable sections for plugins with parameters", () => {
      renderModal([
        { source: "github:owner/plugin1", parameters: { key1: "val1" } },
        { source: "github:owner/plugin2", parameters: { key2: "val2" } },
      ]);

      expect(screen.getByTestId("plugin-section-0")).toBeInTheDocument();
      expect(screen.getByTestId("plugin-section-1")).toBeInTheDocument();
    });

    it("should maintain separate state for each plugin's parameters", async () => {
      const user = userEvent.setup();
      renderModal([
        { source: "github:owner/plugin1", parameters: { key1: "val1" } },
        { source: "github:owner/plugin2", parameters: { key2: "val2" } },
      ]);

      // Update first plugin's parameter
      const input1 = screen.getByTestId("plugin-0-param-key1");
      await user.clear(input1);
      await user.type(input1, "new-val1");

      // Second plugin's parameter should be unchanged
      const input2 = screen.getByTestId("plugin-1-param-key2");
      expect(input2).toHaveValue("val2");

      // First plugin should have new value
      expect(input1).toHaveValue("new-val1");
    });
  });

  // The modal's source line is what the trust checkbox names: "I trust this
  // skill from {sources} with the agent secrets defined in my account." If a
  // source can be displayed as a repo coordinate it does not own, that
  // sentence is asking the user to trust the wrong party.
  describe("Source Display", () => {
    it("shows the whole URL when the host only looks like GitHub", () => {
      const spoofed =
        "https://evil.example/github.com/All-Hands-AI/openhands-plugin";
      renderModal([{ source: spoofed }]);

      expect(screen.getByText(spoofed)).toBeInTheDocument();
      expect(
        screen.queryByText("All-Hands-AI/openhands-plugin"),
      ).not.toBeInTheDocument();
    });

    it("shows the whole URL for a host that merely starts with github.com", () => {
      const spoofed = "https://github.com.evil.example/All-Hands-AI/plugin";
      renderModal([{ source: spoofed }]);

      expect(screen.getByText(spoofed)).toBeInTheDocument();
      expect(screen.queryByText("All-Hands-AI/plugin")).not.toBeInTheDocument();
    });

    it("shortens a real github.com URL to its repo coordinate", () => {
      renderModal([
        { source: "https://github.com/All-Hands-AI/openhands-plugin.git" },
      ]);

      expect(
        screen.getByText("All-Hands-AI/openhands-plugin"),
      ).toBeInTheDocument();
    });

    it("shortens a github: coordinate", () => {
      renderModal([{ source: "github:All-Hands-AI/openhands-plugin" }]);

      // The display name and the source line both read as the coordinate
      // here, so match all of them rather than expecting exactly one.
      expect(
        screen.getAllByText("All-Hands-AI/openhands-plugin").length,
      ).toBeGreaterThan(0);
      expect(
        screen.queryByText("github:All-Hands-AI/openhands-plugin"),
      ).not.toBeInTheDocument();
    });

    it("shows a source that is not a URL unchanged", () => {
      renderModal([{ source: "git@github.com:All-Hands-AI/plugin.git" }]);

      expect(
        screen.getByText("git@github.com:All-Hands-AI/plugin.git"),
      ).toBeInTheDocument();
    });
  });
});
