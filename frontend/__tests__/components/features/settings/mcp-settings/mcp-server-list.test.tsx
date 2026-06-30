import { render, screen } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { MCPServerList } from "#/components/features/settings/mcp-settings/mcp-server-list";
import { useMcpServerHealth } from "#/hooks/query/use-mcp-server-health";

// Mock react-i18next
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
  useMcpServerHealth: vi.fn(() => ({ data: undefined })),
}));

vi.mock("#/hooks/query/use-mcp-test-run", () => ({
  useMcpTestRun: () => ({ data: undefined }),
}));

function renderList(ui: React.ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });

  return render(ui, {
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
}

beforeEach(() => {
  vi.mocked(useMcpServerHealth).mockReturnValue({
    data: undefined,
  } as ReturnType<typeof useMcpServerHealth>);
});

const mockServers = [
  {
    id: "sse-0",
    type: "sse" as const,
    url: "https://very-long-url-that-could-cause-layout-overflow.example.com/api/v1/mcp/server/endpoint/with/many/path/segments",
  },
  {
    id: "stdio-0",
    type: "stdio" as const,
    name: "test-stdio-server",
    command: "python",
    args: ["-m", "test_server"],
  },
];

describe("MCPServerList", () => {
  it("should render servers with proper layout structure", () => {
    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();

    renderList(
      <MCPServerList
        servers={mockServers}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    // Check that the table structure is rendered
    const table = screen.getByRole("table");
    expect(table).toBeInTheDocument();
    expect(table).toHaveClass("w-full");

    // Check that server items are rendered
    const serverItems = screen.getAllByTestId("mcp-server-item");
    expect(serverItems).toHaveLength(2);

    // Check that action buttons are present for each server
    const editButtons = screen.getAllByTestId("edit-mcp-server-button");
    const deleteButtons = screen.getAllByTestId("delete-mcp-server-button");
    expect(editButtons).toHaveLength(2);
    expect(deleteButtons).toHaveLength(2);
  });

  it("should render empty state when no servers", () => {
    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();

    renderList(
      <MCPServerList
        servers={[]}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    expect(screen.getByText("SETTINGS$MCP_NO_SERVERS")).toBeInTheDocument();
  });

  it("should handle long URLs without breaking layout", () => {
    const longUrlServer = {
      id: "sse-0",
      type: "sse" as const,
      url: "https://extremely-long-url-that-would-previously-cause-layout-overflow-and-push-action-buttons-out-of-view.example.com/api/v1/mcp/server/endpoint/with/many/path/segments/and/query/parameters?param1=value1&param2=value2&param3=value3",
    };

    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();

    renderList(
      <MCPServerList
        servers={[longUrlServer]}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    // Check that action buttons are still present and accessible
    const editButton = screen.getByTestId("edit-mcp-server-button");
    const deleteButton = screen.getByTestId("delete-mcp-server-button");

    expect(editButton).toBeInTheDocument();
    expect(deleteButton).toBeInTheDocument();

    // Check that the URL is properly displayed with title attribute for accessibility
    const detailsCells = screen.getAllByTitle(longUrlServer.url);
    expect(detailsCells).toHaveLength(2); // Name and Details columns both have the URL

    // Check that both name and details cells use truncation and have title for tooltip
    const [nameCell, detailsCell] = detailsCells;
    expect(nameCell).toHaveClass("truncate");
    expect(detailsCell).toHaveClass("truncate");
  });

  it("should display command and arguments for STDIO servers", () => {
    const stdioServer = {
      id: "stdio-1",
      type: "stdio" as const,
      name: "test-server",
      command: "python",
      args: ["-m", "test_module", "--verbose"],
    };

    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();

    renderList(
      <MCPServerList
        servers={[stdioServer]}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    // Check that the server details show command + arguments
    const expectedDetails = "python -m test_module --verbose";
    expect(screen.getByTitle(expectedDetails)).toBeInTheDocument();
    expect(screen.getByText(expectedDetails)).toBeInTheDocument();
  });

  it("should fallback to server name for STDIO servers without command", () => {
    const stdioServer = {
      id: "stdio-2",
      type: "stdio" as const,
      name: "fallback-server",
    };

    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();

    renderList(
      <MCPServerList
        servers={[stdioServer]}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    // Check that the server details show the server name as fallback
    // Both name and details columns will have the same value, so we expect 2 elements
    const fallbackElements = screen.getAllByTitle("fallback-server");
    expect(fallbackElements).toHaveLength(2);

    const fallbackTextElements = screen.getAllByText("fallback-server");
    expect(fallbackTextElements).toHaveLength(2);
  });

  it("should display last tested time from the latest test run", () => {
    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();
    const testedAt = new Date(Date.now() - 60_000).toISOString();

    vi.mocked(useMcpServerHealth).mockReturnValue({
      data: {
        server_id: "https://example.com",
        status: "healthy",
        tested_at: testedAt,
      },
    } as ReturnType<typeof useMcpServerHealth>);

    renderList(
      <MCPServerList
        servers={[
          {
            id: "sse-0",
            type: "sse",
            url: "https://example.com",
          },
        ]}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    expect(screen.getByTestId("mcp-server-last-tested")).toHaveTextContent(
      "1m CONVERSATION$AGO",
    );
  });

  it("should show a placeholder when no test run exists", () => {
    const mockOnEdit = vi.fn();
    const mockOnDelete = vi.fn();

    vi.mocked(useMcpServerHealth).mockReturnValue({
      data: {
        server_id: "https://example.com",
        status: "unknown",
      },
    } as ReturnType<typeof useMcpServerHealth>);

    renderList(
      <MCPServerList
        servers={[
          {
            id: "sse-0",
            type: "sse",
            url: "https://example.com",
          },
        ]}
        onEdit={mockOnEdit}
        onDelete={mockOnDelete}
      />,
    );

    expect(screen.getByTestId("mcp-server-last-tested")).toHaveTextContent("—");
  });
});
