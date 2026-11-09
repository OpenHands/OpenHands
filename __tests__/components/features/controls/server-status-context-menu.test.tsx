import { screen } from "@testing-library/react";
import { renderWithProviders } from "test-utils";
import { describe, expect, it, vi } from "vitest";
import { ServerStatusContextMenu } from "#/components/features/controls/server-status-context-menu";
import { ExecutionStatus } from "#/types/agent-server/core/base/common";

vi.mock("react-i18next", async () => {
  const actual = await vi.importActual("react-i18next");
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
      i18n: { changeLanguage: () => Promise.resolve() },
    }),
  };
});

describe("ServerStatusContextMenu", () => {
  it("shows Stop Runtime whenever the caller allows it, even with a paused agent", () => {
    renderWithProviders(
      <ServerStatusContextMenu
        onClose={vi.fn()}
        onStopServer={vi.fn()}
        onStartServer={vi.fn()}
        executionStatus={ExecutionStatus.PAUSED}
      />,
    );

    expect(screen.getByTestId("stop-server-button")).toBeInTheDocument();
    expect(screen.getByTestId("start-server-button")).toBeInTheDocument();
  });

  it("hides Stop Runtime when the caller does not allow it", () => {
    renderWithProviders(
      <ServerStatusContextMenu
        onClose={vi.fn()}
        onStartServer={vi.fn()}
        executionStatus={ExecutionStatus.PAUSED}
      />,
    );

    expect(screen.queryByTestId("stop-server-button")).not.toBeInTheDocument();
    expect(screen.getByTestId("start-server-button")).toBeInTheDocument();
  });
});
