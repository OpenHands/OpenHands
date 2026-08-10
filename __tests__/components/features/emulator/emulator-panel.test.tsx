import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { EmulatorPanel } from "#/components/features/emulator/emulator-panel";
import { EmulatorService } from "#/api/integrations/emulator-service";
import MobileArtifactsService, {
  validateApkFile,
} from "#/api/pentest/mobile-artifacts-service";

vi.mock("#/api/integrations/emulator-service", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("#/api/integrations/emulator-service")
    >();
  return {
    ...actual,
    EmulatorService: {
      getStatus: vi.fn(),
      start: vi.fn(),
      iframePath: () => "/api/emulator/",
    },
  };
});

vi.mock("#/api/pentest/mobile-artifacts-service", async (importOriginal) => {
  const actual =
    await importOriginal<
      typeof import("#/api/pentest/mobile-artifacts-service")
    >();
  return {
    ...actual,
    default: {
      listArtifacts: vi.fn(),
      uploadApk: vi.fn(),
      installArtifact: vi.fn(),
    },
  };
});

vi.mock("#/hooks/use-conversation-id", () => ({
  useConversationId: () => ({ conversationId: "conv-1" }),
}));

vi.mock("#/api/conversation-metadata-store", () => ({
  getStoredConversationMetadata: () => ({ engagement_id: "eng-1" }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

function renderPanel() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <EmulatorPanel />
    </QueryClientProvider>,
  );
}

describe("EmulatorPanel", () => {
  beforeEach(() => {
    vi.mocked(EmulatorService.getStatus).mockReset();
    vi.mocked(EmulatorService.start).mockReset();
    vi.mocked(MobileArtifactsService.listArtifacts).mockReset();
    vi.mocked(MobileArtifactsService.uploadApk).mockReset();
    vi.mocked(MobileArtifactsService.listArtifacts).mockResolvedValue({
      items: [],
    });
  });

  it("shows unavailable without start CTA (AC-192-3)", async () => {
    vi.mocked(EmulatorService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: true,
      url: "/api/emulator/",
    });

    renderPanel();

    expect(await screen.findByTestId("emulator-unavailable")).toBeInTheDocument();
    expect(
      screen.getByTestId("emulator-status-message"),
    ).toHaveTextContent("EMULATOR$UNAVAILABLE");
    expect(
      screen.queryByTestId("emulator-start-button"),
    ).not.toBeInTheDocument();
  });

  it("starts emulator and renders iframe (AC-192-1)", async () => {
    const user = userEvent.setup();
    vi.mocked(EmulatorService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: false,
      url: "/api/emulator/",
    });
    vi.mocked(EmulatorService.start).mockResolvedValue({
      ready: true,
      starting: false,
      unavailable: false,
      url: "/api/emulator/",
    });

    renderPanel();
    await user.click(await screen.findByTestId("emulator-start-button"));

    await waitFor(() => {
      expect(screen.getByTestId("emulator-iframe")).toHaveAttribute(
        "src",
        "/api/emulator/",
      );
    });
    expect(EmulatorService.start).toHaveBeenCalledTimes(1);
  });

  it("rejects IPA client-side with zero POST (AC-192-5)", async () => {
    vi.mocked(EmulatorService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: true,
      url: "/api/emulator/",
    });

    renderPanel();
    const dropzone = await screen.findByTestId("emulator-apk-dropzone");

    const file = new File(["ipa"], "app.ipa", {
      type: "application/octet-stream",
    });
    // Bypass input[accept] filtering — drag/drop is the real IPA path.
    fireEvent.drop(dropzone, {
      dataTransfer: { files: [file] },
    });

    expect(await screen.findByTestId("emulator-upload-error")).toHaveTextContent(
      "EMULATOR$UPLOAD_REJECT_IPA",
    );
    expect(MobileArtifactsService.uploadApk).not.toHaveBeenCalled();
  });

  it("uploads APK and lists artifact (AC-192-4)", async () => {
    const user = userEvent.setup();
    vi.mocked(EmulatorService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: false,
      url: "/api/emulator/",
    });
    vi.mocked(MobileArtifactsService.uploadApk).mockResolvedValue({
      artifact_id: "a1",
      path: "mobile/eng-1/app.apk",
      filename: "app.apk",
      scan_status: "queued",
    });
    vi.mocked(MobileArtifactsService.listArtifacts)
      .mockResolvedValueOnce({ items: [] })
      .mockResolvedValue({
        items: [
          {
            artifact_id: "a1",
            filename: "app.apk",
            scan_status: "queued",
          },
        ],
      });

    renderPanel();
    await screen.findByTestId("emulator-start-button");

    const file = new File(["apk"], "app.apk", {
      type: "application/vnd.android.package-archive",
    });
    await user.upload(screen.getByTestId("emulator-apk-input"), file);

    await waitFor(() => {
      expect(MobileArtifactsService.uploadApk).toHaveBeenCalledTimes(1);
    });
    expect(await screen.findByText("app.apk")).toBeInTheDocument();
  });

  it("allows opening artifacts rail while live (D-192-1)", async () => {
    const user = userEvent.setup();
    vi.mocked(EmulatorService.getStatus).mockResolvedValue({
      ready: true,
      starting: false,
      unavailable: false,
      url: "/api/emulator/",
    });

    renderPanel();
    await screen.findByTestId("emulator-iframe");

    const rail = screen.getByTestId("emulator-artifacts-rail");
    expect(rail).not.toHaveAttribute("open");

    await user.click(rail.querySelector("summary")!);

    expect(rail).toHaveAttribute("open");
    expect(screen.getByTestId("emulator-apk-dropzone")).toBeVisible();
  });

  it("focuses start CTA once on idle (D-192-2)", async () => {
    vi.mocked(EmulatorService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: false,
      url: "/api/emulator/",
    });

    renderPanel();
    const start = await screen.findByTestId("emulator-start-button");
    await waitFor(() => {
      expect(start).toHaveFocus();
    });
  });
});

describe("validateApkFile", () => {
  it("accepts apk and rejects ipa/type/size", () => {
    expect(
      validateApkFile(
        new File(["x"], "a.apk", {
          type: "application/vnd.android.package-archive",
        }),
      ),
    ).toEqual({ ok: true, filename: "a.apk" });
    expect(
      validateApkFile(new File(["x"], "a.ipa", { type: "application/zip" })),
    ).toEqual({ ok: false, reason: "ipa" });
    expect(
      validateApkFile(new File(["x"], "a.txt", { type: "text/plain" })),
    ).toEqual({ ok: false, reason: "type" });
  });
});
