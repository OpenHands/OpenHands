import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { PluginFilesSection } from "#/components/features/plugins/plugin-files-section";
import PluginsService from "#/api/plugins-service";

function renderFilesSection(files: string[] = ["README.md"]) {
  return render(<PluginFilesSection basePath="/plugins/demo" files={files} />, {
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
}

describe("PluginFilesSection", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("fetches and shows a file's content when the file is clicked", async () => {
    const user = userEvent.setup();
    const contentSpy = vi
      .spyOn(PluginsService, "getPluginFileContent")
      .mockResolvedValue({ kind: "text", text: "# Demo readme" });

    renderFilesSection();
    await user.click(screen.getByTestId("file-tree-file-README.md"));

    await waitFor(() =>
      expect(screen.getByTestId("plugin-file-content")).toHaveTextContent(
        "Demo readme",
      ),
    );
    expect(contentSpy).toHaveBeenCalledWith("/plugins/demo", "README.md");
  });

  it("closes the content viewer when the selected file is clicked again", async () => {
    const user = userEvent.setup();
    vi.spyOn(PluginsService, "getPluginFileContent").mockResolvedValue({
      kind: "text",
      text: "# Demo readme",
    });

    renderFilesSection();
    await user.click(screen.getByTestId("file-tree-file-README.md"));
    await screen.findByTestId("plugin-file-content");
    await user.click(screen.getByTestId("file-tree-file-README.md"));

    expect(screen.queryByTestId("plugin-file-content")).not.toBeInTheDocument();
  });

  it("shows a binary notice instead of content for binary files", async () => {
    const user = userEvent.setup();
    vi.spyOn(PluginsService, "getPluginFileContent").mockResolvedValue({
      kind: "binary",
      text: null,
    });

    renderFilesSection(["logo.png"]);
    await user.click(screen.getByTestId("file-tree-file-logo.png"));

    expect(
      await screen.findByText("FILES$BINARY_FALLBACK"),
    ).toBeInTheDocument();
  });

  it("shows a load error when fetching the file fails", async () => {
    const user = userEvent.setup();
    vi.spyOn(PluginsService, "getPluginFileContent").mockRejectedValue(
      new Error("unreachable"),
    );

    renderFilesSection();
    await user.click(screen.getByTestId("file-tree-file-README.md"));

    expect(await screen.findByText("FILES$LOAD_ERROR")).toBeInTheDocument();
  });
});
