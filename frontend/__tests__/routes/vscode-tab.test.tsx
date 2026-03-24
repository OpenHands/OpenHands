import { screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";
import VSCodeTab from "#/routes/vscode-tab";
import { useUnifiedVSCodeUrl } from "#/hooks/query/use-unified-vscode-url";
import { useAgentState } from "#/hooks/use-agent-state";
import { AgentState } from "#/types/agent-state";

vi.mock("#/hooks/query/use-unified-vscode-url");
vi.mock("#/hooks/use-agent-state");
vi.mock("#/utils/feature-flags", () => ({
  VSCODE_IN_NEW_TAB: () => false,
}));

function mockVSCodeUrlHook(
  value: Partial<ReturnType<typeof useUnifiedVSCodeUrl>>,
) {
  vi.mocked(useUnifiedVSCodeUrl).mockReturnValue({
    data: { url: "http://localhost:3000/vscode", error: null },
    error: null,
    isLoading: false,
    isError: false,
    isSuccess: true,
    status: "success",
    refetch: vi.fn(),
    ...value,
  } as ReturnType<typeof useUnifiedVSCodeUrl>);
}

describe("VSCodeTab", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("keeps VSCode accessible when the agent is in an error state", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.ERROR,
    });
    mockVSCodeUrlHook({});

    renderWithProviders(<VSCodeTab />);

    expect(
      screen.queryByText("DIFF_VIEWER$WAITING_FOR_RUNTIME"),
    ).not.toBeInTheDocument();
    expect(screen.getByTitle("VSCODE$TITLE")).toHaveAttribute(
      "src",
      "http://localhost:3000/vscode",
    );
  });

  it("still waits while the runtime is starting", () => {
    vi.mocked(useAgentState).mockReturnValue({
      curAgentState: AgentState.LOADING,
    });
    mockVSCodeUrlHook({});

    renderWithProviders(<VSCodeTab />);

    expect(
      screen.getByText("DIFF_VIEWER$WAITING_FOR_RUNTIME"),
    ).toBeInTheDocument();
    expect(screen.queryByTitle("VSCODE$TITLE")).not.toBeInTheDocument();
  });
});
