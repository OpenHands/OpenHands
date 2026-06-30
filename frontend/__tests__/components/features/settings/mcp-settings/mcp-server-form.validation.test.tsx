import { render, screen, fireEvent } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi } from "vitest";
import { MCPServerForm } from "#/components/features/settings/mcp-settings/mcp-server-form";

function renderForm(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

// i18n mock
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

vi.mock("#/hooks/mutation/use-test-mcp-server", () => ({
  useTestMcpServer: () => ({
    mutate: vi.fn(),
    isPending: false,
  }),
}));

vi.mock("#/hooks/query/use-mcp-server-health", () => ({
  useMcpServerHealth: () => ({ data: undefined }),
}));

vi.mock("#/hooks/query/use-mcp-test-run", () => ({
  useMcpTestRun: () => ({ data: undefined }),
}));

describe("MCPServerForm validation", () => {
  const noop = () => {};

  it("rejects invalid env var lines and allows blank lines", () => {
    const onSubmit = vi.fn();

    renderForm(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "stdio" }}
        existingServers={[]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    // Fill required fields
    fireEvent.change(screen.getByTestId("name-input"), {
      target: { value: "my-server" },
    });
    fireEvent.change(screen.getByTestId("command-input"), {
      target: { value: "npx" },
    });

    // Invalid env entries mixed with blank lines
    fireEvent.change(screen.getByTestId("env-input"), {
      target: { value: "invalid\n\nKEY=value\n=novalue\nKEY_ONLY=" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    // Should show invalid env format error
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_ENV_INVALID_FORMAT"),
    ).toBeInTheDocument();

    // Fix env with valid lines and blank lines
    fireEvent.change(screen.getByTestId("env-input"), {
      target: { value: "KEY=value\n\nANOTHER=123" },
    });

    fireEvent.click(screen.getByTestId("submit-button"));

    // No error; submit should be called
    expect(onSubmit).toHaveBeenCalledTimes(1);
  });

  it("rejects duplicate URLs across sse/shttp types", () => {
    const onSubmit = vi.fn();

    const existingServers = [
      { id: "sse-1", type: "sse" as const, url: "https://api.example.com" },
      { id: "shttp-1", type: "shttp" as const, url: "https://x.example.com" },
    ];

    const r1 = renderForm(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp", type: "sse" }}
        existingServers={existingServers}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getAllByTestId("url-input")[0], {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getAllByTestId("submit-button")[0]);
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_URL_DUPLICATE"),
    ).toBeInTheDocument();

    // Unmount first form, then check shttp duplicate
    r1.unmount();

    const r2 = renderForm(
      <MCPServerForm
        mode="add"
        server={{ id: "tmp2", type: "shttp" }}
        existingServers={existingServers}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    fireEvent.change(screen.getAllByTestId("url-input")[0], {
      target: { value: "https://api.example.com" },
    });

    fireEvent.click(screen.getAllByTestId("submit-button")[0]);
    expect(
      screen.getByText("SETTINGS$MCP_ERROR_URL_DUPLICATE"),
    ).toBeInTheDocument();

    r2.unmount();
  });

  it("disables save on edit when the form is unchanged", () => {
    const onSubmit = vi.fn();
    const existingServer = {
      id: "sse-0",
      type: "sse" as const,
      name: "myserver",
      url: "https://api.example.com",
      api_key: "secret",
    };

    renderForm(
      <MCPServerForm
        mode="edit"
        server={existingServer}
        existingServers={[existingServer]}
        onSubmit={onSubmit}
        onCancel={noop}
      />,
    );

    expect(screen.getByTestId("submit-button")).toBeDisabled();

    fireEvent.change(screen.getByTestId("url-input"), {
      target: { value: "https://api.example.com/changed" },
    });

    expect(screen.getByTestId("submit-button")).not.toBeDisabled();
  });
});
