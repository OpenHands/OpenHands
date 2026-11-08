import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { HttpError } from "@openhands/typescript-client";
import { I18nKey } from "#/i18n/declaration";
import { ScriptBody } from "#/components/features/automations/detail/script-section";
import AutomationService from "#/api/automation-service/automation-service.api";
import {
  __resetActiveStoreForTests,
  setActiveSelection,
  setRegisteredBackends,
} from "#/api/backend-registry/active-store";
import { ActiveBackendProvider } from "#/contexts/active-backend-context";
import type { Automation } from "#/types/automation";
import { packTar, packTarGzip } from "#/utils/tar-gzip";

const automation: Automation = {
  id: "auto-1",
  name: "Nightly scan",
  prompt: null,
  trigger: { type: "cron" },
  enabled: true,
  created_at: "2026-09-01T00:00:00Z",
  updated_at: "2026-09-02T00:00:00Z",
  entrypoint: "python main.py",
};

function renderSection(subject: Automation = automation) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ActiveBackendProvider>
        <ScriptBody automation={subject} />
      </ActiveBackendProvider>
    </QueryClientProvider>,
  );
}

function filePaths(): string[] {
  return screen
    .getAllByTestId("automation-script-file")
    .map((row) => row.querySelector(".truncate")?.textContent ?? "");
}

beforeEach(() => {
  __resetActiveStoreForTests();
  setRegisteredBackends([
    {
      id: "local-1",
      name: "Local",
      host: "http://localhost:8000",
      apiKey: "k",
      kind: "local",
    },
  ]);
  setActiveSelection({ backendId: "local-1" });
});

afterEach(() => {
  __resetActiveStoreForTests();
  vi.restoreAllMocks();
});

describe("ScriptBody", () => {
  it("lists the bundle's files with the entrypoint first and badged", async () => {
    // Arrange: macOS tar adds an AppleDouble twin for every file.
    vi.spyOn(AutomationService, "fetchTarballBytes").mockResolvedValue(
      new Uint8Array(
        await packTarGzip([
          { name: "lib/util.py", content: "x = 1\n" },
          { name: "main.py", content: "print('hi')\n" },
          { name: "._main.py", content: "metadata" },
          { name: "config.json", content: "{}\n" },
        ]),
      ),
    );

    // Act
    renderSection();

    // Assert
    expect(await screen.findAllByTestId("automation-script-file")).toHaveLength(
      3,
    );
    expect(filePaths()).toEqual(["main.py", "config.json", "lib/util.py"]);
    expect(
      screen.getByTestId("automation-script-entrypoint"),
    ).toHaveTextContent("python main.py");
    expect(
      screen.getAllByText(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_ENTRYPOINT),
    ).toHaveLength(1);
    expect(AutomationService.fetchTarballBytes).toHaveBeenCalledWith("auto-1");
  });

  it("keeps each file's source closed until it is opened", async () => {
    // Arrange
    vi.spyOn(AutomationService, "fetchTarballBytes").mockResolvedValue(
      new Uint8Array(
        await packTarGzip([
          { name: "main.py", content: "print('hi')\n" },
          { name: "config.json", content: "{}\n" },
        ]),
      ),
    );
    const user = userEvent.setup();
    renderSection();
    const [mainFile, configFile] = await screen.findAllByTestId(
      "automation-script-file",
    );
    const mainToggle = screen.getByRole("button", { name: /main\.py/ });
    const configToggle = screen.getByRole("button", { name: /config\.json/ });
    expect(mainFile).not.toHaveTextContent("print('hi')");
    expect(configFile).not.toHaveTextContent("{}");
    expect(mainToggle).toHaveAttribute("aria-expanded", "false");
    expect(configToggle).toHaveAttribute("aria-expanded", "false");

    // Act
    await user.click(mainToggle);

    // Assert
    expect(mainFile).toHaveTextContent("print('hi')");
    expect(mainToggle).toHaveAttribute("aria-expanded", "true");
    expect(configFile).not.toHaveTextContent("{}");
    expect(configToggle).toHaveAttribute("aria-expanded", "false");
  });

  it("shows a loading message until the bundle arrives", () => {
    // Arrange
    vi.spyOn(AutomationService, "fetchTarballBytes").mockReturnValue(
      new Promise(() => {}),
    );

    // Act
    renderSection();

    // Assert
    expect(
      screen.getByText(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_LOADING),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-script-unavailable"),
    ).not.toBeInTheDocument();
  });

  it("points at the tarball download when the service cannot hand the bundle back", async () => {
    // Arrange: an externally hosted bundle the service refuses to proxy.
    vi.spyOn(AutomationService, "fetchTarballBytes").mockRejectedValue(
      new HttpError(422, "Unprocessable Entity", {
        detail: "Cannot proxy s3:// tarball URLs",
      }),
    );

    // Act
    renderSection();

    // Assert
    expect(
      await screen.findByTestId("automation-script-unavailable"),
    ).toBeInTheDocument();
    expect(screen.queryAllByTestId("automation-script-file")).toHaveLength(0);
  });

  it("points at the tarball download when the bytes are not an archive", async () => {
    // Arrange
    vi.spyOn(AutomationService, "fetchTarballBytes").mockResolvedValue(
      new Uint8Array(new TextEncoder().encode("<html>".padEnd(1024, "x"))),
    );

    // Act
    renderSection();

    // Assert
    expect(
      await screen.findByTestId("automation-script-unavailable"),
    ).toBeInTheDocument();
  });

  it("labels a binary member instead of rendering it", async () => {
    // Arrange: a member whose bytes are not UTF-8.
    const archive = packTar([{ name: "model.bin", content: "AAAA" }]);
    archive.fill(0xff, 512, 516);
    vi.spyOn(AutomationService, "fetchTarballBytes").mockResolvedValue(archive);

    // Act
    renderSection({ ...automation, entrypoint: undefined });

    // Assert
    expect(
      await screen.findByText(I18nKey.AUTOMATIONS$DETAIL$SCRIPT_BINARY_FILE),
    ).toBeInTheDocument();
    expect(
      screen.queryByTestId("automation-script-entrypoint"),
    ).not.toBeInTheDocument();
  });
});
