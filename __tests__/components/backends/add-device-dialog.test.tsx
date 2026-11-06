import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { renderWithProviders } from "test-utils";

import type { MarsBridge } from "#/api/mars/mars-tunnel-backend";
import AddDeviceDialog from "#/components/features/backends/add-device-dialog";

const inMinutes = (minutes: number) =>
  new Date(Date.now() + minutes * 60_000).toISOString();

function withPairing(
  ...replies: { links: string[]; expiresAt: string }[]
): ReturnType<typeof vi.fn> {
  const createPairingLink = vi.fn();
  for (const reply of replies) createPairingLink.mockResolvedValueOnce(reply);
  window.marsBridge = { createPairingLink } as unknown as MarsBridge;
  return createPairingLink;
}

afterEach(() => {
  delete window.marsBridge;
});

describe("AddDeviceDialog", () => {
  it("shows the link as a QR code, copies it, and warns about plain HTTP", async () => {
    const user = userEvent.setup();
    withPairing({
      links: [
        "http://192.168.1.20:8000/pair#s1",
        "http://100.101.1.2:8000/pair#s1",
      ],
      expiresAt: inMinutes(5),
    });
    renderWithProviders(<AddDeviceDialog onClose={vi.fn()} />);

    expect(await screen.findByTestId("add-device-qr")).toBeInTheDocument();
    expect(screen.getByTestId("add-device-plain-http")).toBeInTheDocument();
    await user.selectOptions(screen.getByTestId("add-device-address"), "1");
    await user.click(screen.getByTestId("add-device-copy"));

    expect(await navigator.clipboard.readText()).toBe(
      "http://100.101.1.2:8000/pair#s1",
    );
  });

  it("selects the link and says so when this page cannot copy, as on plain HTTP", async () => {
    const user = userEvent.setup();
    // Browsers leave `navigator.clipboard` out on an insecure page.
    const clipboard = Object.getOwnPropertyDescriptor(navigator, "clipboard");
    Object.defineProperty(navigator, "clipboard", {
      value: undefined,
      configurable: true,
    });
    try {
      withPairing({
        links: ["http://192.168.1.20:8000/pair#s1"],
        expiresAt: inMinutes(5),
      });
      renderWithProviders(<AddDeviceDialog onClose={vi.fn()} />);

      const field =
        await screen.findByTestId<HTMLInputElement>("add-device-link");
      expect(field.value).toBe("http://192.168.1.20:8000/pair#s1");
      await user.click(screen.getByTestId("add-device-copy"));

      expect(screen.getByTestId("add-device-copy-failed")).toBeInTheDocument();
      expect(document.activeElement).toBe(field);
      expect(field.selectionStart).toBe(0);
      expect(field.selectionEnd).toBe(field.value.length);
    } finally {
      if (clipboard) Object.defineProperty(navigator, "clipboard", clipboard);
    }
  });

  it("does not warn when the link is HTTPS", async () => {
    withPairing({
      links: ["https://studio.tailnet.ts.net/pair#s1"],
      expiresAt: inMinutes(5),
    });
    renderWithProviders(<AddDeviceDialog onClose={vi.fn()} />);

    expect(await screen.findByTestId("add-device-qr")).toBeInTheDocument();
    expect(screen.queryByTestId("add-device-plain-http")).toBeNull();
  });

  it("explains that a Canvas listening only on this computer cannot add devices", async () => {
    withPairing({ links: [], expiresAt: inMinutes(5) });
    renderWithProviders(<AddDeviceDialog onClose={vi.fn()} />);

    expect(
      await screen.findByTestId("add-device-loopback-only"),
    ).toBeInTheDocument();
    expect(screen.queryByTestId("add-device-qr")).toBeNull();
  });

  it("replaces an expired code with a button for a new link", async () => {
    const user = userEvent.setup();
    const createPairingLink = withPairing(
      {
        links: ["http://192.168.1.20:8000/pair#old"],
        expiresAt: inMinutes(-1),
      },
      { links: ["http://192.168.1.20:8000/pair#new"], expiresAt: inMinutes(5) },
    );
    renderWithProviders(<AddDeviceDialog onClose={vi.fn()} />);

    await user.click(await screen.findByTestId("add-device-new-link"));

    expect(await screen.findByTestId("add-device-qr")).toBeInTheDocument();
    await waitFor(() => expect(createPairingLink).toHaveBeenCalledTimes(2));
  });
});
