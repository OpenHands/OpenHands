import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { DesktopPanel } from "#/components/features/desktop/desktop-panel";
import { DesktopService } from "#/api/integrations/desktop-service";

vi.mock("#/api/integrations/desktop-service", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("#/api/integrations/desktop-service")>();
  return {
    ...actual,
    DesktopService: {
      getStatus: vi.fn(),
      start: vi.fn(),
      iframePath: () => "/api/desktop/index.html",
    },
  };
});

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

describe("DesktopPanel", () => {
  beforeEach(() => {
    vi.mocked(DesktopService.getStatus).mockReset();
    vi.mocked(DesktopService.start).mockReset();
  });

  it("shows unavailable state when desktop scripts are missing", async () => {
    vi.mocked(DesktopService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: true,
      url: "/api/desktop/",
    });

    render(<DesktopPanel />);

    expect(
      await screen.findByTestId("desktop-status-message"),
    ).toHaveTextContent("DESKTOP$UNAVAILABLE");
    expect(screen.queryByTestId("desktop-open-button")).not.toBeInTheDocument();
  });

  it("shows unavailable when start fails because the proxy is missing", async () => {
    const { DesktopRequestError } = await import(
      "#/api/integrations/desktop-service"
    );
    vi.mocked(DesktopService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: false,
      url: "/api/desktop/",
    });
    vi.mocked(DesktopService.start).mockRejectedValue(
      new DesktopRequestError("not found", {
        status: 404,
        unavailable: true,
      }),
    );

    const user = userEvent.setup();
    render(<DesktopPanel />);
    await user.click(await screen.findByTestId("desktop-open-button"));

    expect(
      await screen.findByTestId("desktop-status-message"),
    ).toHaveTextContent("DESKTOP$UNAVAILABLE");
    expect(screen.queryByTestId("desktop-open-button")).not.toBeInTheDocument();
  });

  it("starts the desktop and renders the iframe", async () => {
    const user = userEvent.setup();
    vi.mocked(DesktopService.getStatus).mockResolvedValue({
      ready: false,
      starting: false,
      unavailable: false,
      url: "/api/desktop/index.html",
    });
    vi.mocked(DesktopService.start).mockResolvedValue({
      ready: true,
      starting: false,
      unavailable: false,
      url: "/api/desktop/index.html",
    });

    render(<DesktopPanel />);

    await user.click(await screen.findByTestId("desktop-open-button"));

    await waitFor(() => {
      expect(screen.getByTestId("desktop-iframe")).toHaveAttribute(
        "src",
        "/api/desktop/index.html",
      );
    });
    expect(DesktopService.start).toHaveBeenCalledTimes(1);
  });
});
