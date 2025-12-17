import { render, screen } from "@testing-library/react";
import { describe, it, expect, vi } from "vitest";
import { MCPServerList } from "#/components/features/settings/mcp-settings/mcp-server-list";

// Mock react-i18next
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

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

    render(
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

    render(
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

    render(
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

    render(
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

    render(
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
});
