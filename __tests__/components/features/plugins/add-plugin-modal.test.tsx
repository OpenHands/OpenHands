import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { AddPluginModal } from "#/components/features/plugins/add-plugin-modal";
import PluginsManagementService from "#/api/plugins-management-service";

function renderAddPluginModal(onClose = vi.fn()) {
  render(<AddPluginModal onClose={onClose} />, {
    wrapper: ({ children }) => (
      <QueryClientProvider
        client={
          new QueryClient({ defaultOptions: { queries: { retry: false } } })
        }
      >
        {children}
      </QueryClientProvider>
    ),
  });
  return { onClose };
}

describe("AddPluginModal", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("installs the plugin with the entered source on submit", async () => {
    const user = userEvent.setup();
    const installSpy = vi
      .spyOn(PluginsManagementService, "installPlugin")
      .mockResolvedValue({
        name: "weather",
        version: "1.0.0",
        description: "",
        enabled: true,
        source: "github:acme/weather",
        resolved_ref: null,
        repo_path: null,
        installed_at: "2026-06-01T00:00:00Z",
        install_path: "/home/.openhands/plugins/installed/weather",
      });

    renderAddPluginModal();
    await user.type(
      screen.getByTestId("add-plugin-source-input"),
      "github:acme/weather",
    );
    await user.click(screen.getByTestId("add-plugin-submit"));

    await waitFor(() =>
      expect(installSpy).toHaveBeenCalledWith({
        source: "github:acme/weather",
        ref: null,
        repo_path: null,
      }),
    );
  });

  it("disables the submit button while the source field is empty", () => {
    renderAddPluginModal();

    expect(screen.getByTestId("add-plugin-submit")).toBeDisabled();
  });
});
