import { render, screen, waitFor } from "@testing-library/react";
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { MemoryRouter, Routes, Route } from "react-router";
import LaunchRoute from "#/routes/launch";

// Mock the hooks
const mockMutateAsync = vi.fn();
const mockNavigate = vi.fn();

vi.mock("#/hooks/mutation/use-create-conversation", () => ({
  useCreateConversation: () => ({
    mutateAsync: mockMutateAsync,
    isPending: false,
  }),
}));

vi.mock("react-router", async () => {
  const actual = await vi.importActual("react-router");
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => ({ data: true }),
}));

vi.mock("#/hooks/query/use-config", () => ({
  useConfig: () => ({ data: { APP_MODE: "saas" } }),
}));

function renderLaunchRoute(searchParams: string) {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
    },
  });

  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[`/launch${searchParams}`]}>
        <Routes>
          <Route path="/launch" element={<LaunchRoute />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

describe("LaunchRoute", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockMutateAsync.mockResolvedValue({ conversation_id: "test-conv-123" });
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe("Query Parameter Parsing", () => {
    it("should parse valid base64 encoded plugins", async () => {
      // Single plugin with parameters
      const plugins = [
        {
          source: "github:owner/repo",
          ref: "main",
          parameters: { apiKey: "test-key", maxRetries: 3 },
        },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      expect(screen.getByTestId("plugin-launch-modal")).toBeInTheDocument();
      expect(screen.getByText("owner/repo")).toBeInTheDocument();
    });

    it("should parse multiple plugins from base64", async () => {
      const plugins = [
        { source: "github:owner/repo1", parameters: { key: "value1" } },
        { source: "github:owner/repo2" },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      expect(screen.getByTestId("plugin-launch-modal")).toBeInTheDocument();
      // Plugin names appear multiple times, use getAllByText
      expect(screen.getAllByText("owner/repo1").length).toBeGreaterThan(0);
      expect(screen.getAllByText("owner/repo2").length).toBeGreaterThan(0);
    });

    it("should show error for invalid base64 encoding", () => {
      renderLaunchRoute("?plugins=not-valid-base64!!!");

      expect(screen.getByTestId("launch-error")).toBeInTheDocument();
      expect(screen.getByText("LAUNCH$ERROR_INVALID_FORMAT")).toBeInTheDocument();
    });

    it("should show error for invalid JSON in decoded base64", () => {
      const invalidJson = btoa("not valid json");

      renderLaunchRoute(`?plugins=${invalidJson}`);

      expect(screen.getByTestId("launch-error")).toBeInTheDocument();
      expect(screen.getByText("LAUNCH$ERROR_INVALID_FORMAT")).toBeInTheDocument();
    });

    it("should show error when decoded plugins is not an array", () => {
      const notArray = btoa(JSON.stringify({ source: "github:owner/repo" }));

      renderLaunchRoute(`?plugins=${notArray}`);

      expect(screen.getByTestId("launch-error")).toBeInTheDocument();
      expect(screen.getByText("LAUNCH$ERROR_INVALID_FORMAT")).toBeInTheDocument();
    });

    it("should show error when plugin is missing source", () => {
      const missingSource = btoa(JSON.stringify([{ ref: "main" }]));

      renderLaunchRoute(`?plugins=${missingSource}`);

      expect(screen.getByTestId("launch-error")).toBeInTheDocument();
      expect(screen.getByText("LAUNCH$ERROR_INVALID_FORMAT")).toBeInTheDocument();
    });

    it("should parse simple params format (plugin_source)", () => {
      renderLaunchRoute("?plugin_source=github:owner/simple-repo");

      expect(screen.getByTestId("plugin-launch-modal")).toBeInTheDocument();
      // Plugin name appears multiple times, use getAllByText
      expect(screen.getAllByText("owner/simple-repo").length).toBeGreaterThan(0);
    });

    it("should parse simple params with ref", () => {
      renderLaunchRoute(
        "?plugin_source=github:owner/repo&plugin_ref=v1.0.0",
      );

      expect(screen.getByTestId("plugin-launch-modal")).toBeInTheDocument();
      // Plugin name appears multiple times, use getAllByText
      expect(screen.getAllByText("owner/repo").length).toBeGreaterThan(0);
    });

    it("should show error when no plugins specified", () => {
      renderLaunchRoute("");

      expect(screen.getByTestId("launch-error")).toBeInTheDocument();
      expect(screen.getByText("LAUNCH$ERROR_NO_PLUGINS")).toBeInTheDocument();
    });
  });

  describe("Message Sanitization", () => {
    it("should display sanitized message", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const message = "Hello, this is a safe message";

      renderLaunchRoute(`?plugins=${encoded}&message=${encodeURIComponent(message)}`);

      expect(screen.getByText("Hello, this is a safe message")).toBeInTheDocument();
    });

    it("should remove script tags from message (XSS prevention)", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const maliciousMessage = '<script>alert("xss")</script>Safe text';

      renderLaunchRoute(
        `?plugins=${encoded}&message=${encodeURIComponent(maliciousMessage)}`,
      );

      // Script tags should be stripped (text content preserved but safe as plain text)
      expect(screen.queryByText(/<script>/)).not.toBeInTheDocument();
      // The text content remains but is rendered as plain text, not executed
      expect(screen.getByText('alert("xss")Safe text')).toBeInTheDocument();
    });

    it("should strip img tags with onerror handlers (XSS prevention)", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const maliciousMessage = '<img src=x onerror=alert(1)>Safe content';

      renderLaunchRoute(
        `?plugins=${encoded}&message=${encodeURIComponent(maliciousMessage)}`,
      );

      // All HTML should be stripped, leaving only safe text
      expect(screen.queryByText(/<img/)).not.toBeInTheDocument();
      expect(screen.queryByText(/onerror/)).not.toBeInTheDocument();
      expect(screen.getByText("Safe content")).toBeInTheDocument();
    });

    it("should strip svg tags with onload handlers (XSS prevention)", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const maliciousMessage = '<svg/onload=alert(1)>Safe content';

      renderLaunchRoute(
        `?plugins=${encoded}&message=${encodeURIComponent(maliciousMessage)}`,
      );

      // All HTML should be stripped
      expect(screen.queryByText(/<svg/)).not.toBeInTheDocument();
      expect(screen.queryByText(/onload/)).not.toBeInTheDocument();
      expect(screen.getByText("Safe content")).toBeInTheDocument();
    });

    it("should strip all HTML tags for plain text display", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const htmlMessage = '<div><p>Hello</p> <b>World</b></div>';

      renderLaunchRoute(
        `?plugins=${encoded}&message=${encodeURIComponent(htmlMessage)}`,
      );

      // HTML tags should be stripped, text preserved
      expect(screen.getByText("Hello World")).toBeInTheDocument();
    });

    it("should truncate long messages", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const longMessage = "A".repeat(600);

      renderLaunchRoute(`?plugins=${encoded}&message=${encodeURIComponent(longMessage)}`);

      // Should be truncated to 500 chars
      const displayedMessage = screen.getByText(/A{100,}/);
      expect(displayedMessage.textContent?.length).toBeLessThanOrEqual(500);
    });
  });

  describe("Plugin Parameter Forms", () => {
    it("should render text input for string parameters", async () => {
      const plugins = [
        {
          source: "github:owner/repo",
          parameters: { apiKey: "initial-value" },
        },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const input = screen.getByTestId("plugin-0-param-apiKey");
      expect(input).toHaveAttribute("type", "text");
      expect(input).toHaveValue("initial-value");
    });

    it("should render number input for number parameters", () => {
      const plugins = [
        {
          source: "github:owner/repo",
          parameters: { maxRetries: 5 },
        },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const input = screen.getByTestId("plugin-0-param-maxRetries");
      expect(input).toHaveAttribute("type", "number");
      expect(input).toHaveValue(5);
    });

    it("should render checkbox for boolean parameters", () => {
      const plugins = [
        {
          source: "github:owner/repo",
          parameters: { debugMode: true },
        },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const checkbox = screen.getByTestId("plugin-0-param-debugMode");
      expect(checkbox).toHaveAttribute("type", "checkbox");
      expect(checkbox).toBeChecked();
    });

    it("should allow editing text parameter values", async () => {
      const user = userEvent.setup();
      const plugins = [
        {
          source: "github:owner/repo",
          parameters: { apiKey: "old-value" },
        },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const input = screen.getByTestId("plugin-0-param-apiKey");
      await user.clear(input);
      await user.type(input, "new-value");

      expect(input).toHaveValue("new-value");
    });
  });

  describe("Modal UI", () => {
    it("should display start conversation button", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      expect(screen.getByTestId("start-conversation-button")).toBeInTheDocument();
    });

    it("should display close button", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      expect(screen.getByTestId("close-button")).toBeInTheDocument();
    });

    it("should navigate home when close button clicked", async () => {
      const user = userEvent.setup();
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      await user.click(screen.getByTestId("close-button"));

      expect(mockNavigate).toHaveBeenCalledWith("/");
    });

    it("should show plugins section for plugins without parameters", () => {
      const plugins = [
        { source: "github:owner/repo-without-params" },
        { source: "github:owner/another-repo" },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      expect(screen.getByText("LAUNCH$PLUGINS")).toBeInTheDocument();
      // Plugin names appear multiple times (title area and list), use getAllByText
      expect(screen.getAllByText("owner/repo-without-params").length).toBeGreaterThan(0);
      expect(screen.getAllByText("owner/another-repo").length).toBeGreaterThan(0);
    });

    it("should show additional plugins label when mixing params and no-params plugins", () => {
      const plugins = [
        { source: "github:owner/with-params", parameters: { key: "value" } },
        { source: "github:owner/without-params" },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      expect(screen.getByText("LAUNCH$ADDITIONAL_PLUGINS")).toBeInTheDocument();
    });
  });

  describe("Trust Checkbox", () => {
    it("should display trust checkbox unchecked by default", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const checkbox = screen.getByTestId("trust-checkbox");
      expect(checkbox).toBeInTheDocument();
      expect(checkbox).not.toBeChecked();
    });

    it("should display trust checkbox with associated label", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // Check that the checkbox has a label associated with it
      const checkbox = screen.getByTestId("trust-checkbox");
      const label = document.querySelector('label[for="trust-checkbox"]');
      expect(label).toBeInTheDocument();
    });

    it("should have trust checkbox label that references the translation key", () => {
      const plugins = [
        { source: "github:owner/repo1" },
        { source: "github:owner/repo2" },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // In test environment, the translation key is shown
      const label = document.querySelector('label[for="trust-checkbox"]');
      expect(label).toBeInTheDocument();
      expect(label?.textContent).toContain("LAUNCH$TRUST_SKILL_CHECKBOX");
    });

    it("should disable start button when trust checkbox is unchecked", () => {
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const button = screen.getByTestId("start-conversation-button");
      expect(button).toBeDisabled();
    });

    it("should enable start button when trust checkbox is checked", async () => {
      const user = userEvent.setup();
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      const checkbox = screen.getByTestId("trust-checkbox");
      await user.click(checkbox);

      const button = screen.getByTestId("start-conversation-button");
      expect(button).not.toBeDisabled();
    });
  });

  describe("Conversation Creation", () => {
    it("should call createConversation with plugins when start button clicked after checking trust", async () => {
      const user = userEvent.setup();
      const plugins = [
        { source: "github:owner/repo", parameters: { apiKey: "test" } },
      ];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // First check the trust checkbox
      await user.click(screen.getByTestId("trust-checkbox"));
      await user.click(screen.getByTestId("start-conversation-button"));

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          plugins: [
            {
              source: "github:owner/repo",
              ref: null,
              repo_path: null,
              parameters: { apiKey: "test" },
            },
          ],
          query: undefined,
        });
      });
    });

    it("should call createConversation with plugins and initial message when message is provided", async () => {
      const user = userEvent.setup();
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));
      const message = "/city-weather:now Tokyo";

      renderLaunchRoute(`?plugins=${encoded}&message=${encodeURIComponent(message)}`);

      // First check the trust checkbox
      await user.click(screen.getByTestId("trust-checkbox"));
      await user.click(screen.getByTestId("start-conversation-button"));

      await waitFor(() => {
        expect(mockMutateAsync).toHaveBeenCalledWith({
          plugins: [
            {
              source: "github:owner/repo",
              ref: null,
              repo_path: null,
              parameters: null,
            },
          ],
          query: "/city-weather:now Tokyo",
        });
      });
    });

    it("should navigate to conversation after successful creation", async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockResolvedValue({ conversation_id: "new-conv-456" });

      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // First check the trust checkbox
      await user.click(screen.getByTestId("trust-checkbox"));
      await user.click(screen.getByTestId("start-conversation-button"));

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith("/conversations/new-conv-456");
      });
    });

    it("should not call createConversation when trust checkbox is not checked", async () => {
      const user = userEvent.setup();
      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // Try to click without checking trust (button should be disabled)
      const button = screen.getByTestId("start-conversation-button");
      expect(button).toBeDisabled();

      // Even attempting to click shouldn't call the mutation
      await user.click(button);

      expect(mockMutateAsync).not.toHaveBeenCalled();
    });
  });

  describe("Error Handling", () => {
    it("should display go home button on error", () => {
      renderLaunchRoute("");

      expect(screen.getByTestId("go-home-button")).toBeInTheDocument();
    });

    it("should navigate home when go home button clicked", async () => {
      const user = userEvent.setup();

      renderLaunchRoute("");

      await user.click(screen.getByTestId("go-home-button"));

      expect(mockNavigate).toHaveBeenCalledWith("/");
    });

    it("should show creation failed error when API call fails", async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue(new Error("API Error"));

      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // First check the trust checkbox
      await user.click(screen.getByTestId("trust-checkbox"));
      await user.click(screen.getByTestId("start-conversation-button"));

      await waitFor(() => {
        expect(screen.getByTestId("launch-error")).toBeInTheDocument();
        expect(screen.getByText("LAUNCH$ERROR_CREATION_FAILED")).toBeInTheDocument();
      });
    });

    it("should show try again button on creation failure", async () => {
      const user = userEvent.setup();
      mockMutateAsync.mockRejectedValue(new Error("API Error"));

      const plugins = [{ source: "github:owner/repo" }];
      const encoded = btoa(JSON.stringify(plugins));

      renderLaunchRoute(`?plugins=${encoded}`);

      // First check the trust checkbox
      await user.click(screen.getByTestId("trust-checkbox"));
      await user.click(screen.getByTestId("start-conversation-button"));

      await waitFor(() => {
        expect(screen.getByTestId("try-again-button")).toBeInTheDocument();
      });
    });
  });
});
