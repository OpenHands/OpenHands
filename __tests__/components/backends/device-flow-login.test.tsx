import React from "react";
import { fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { DeviceFlowAuth } from "#/components/features/backends/device-flow-auth";
import type { UseDeviceFlowReturn } from "#/hooks/use-device-flow";

const useDeviceFlowMock = vi.hoisted(() => vi.fn());
const useTranslationMock = vi.hoisted(() => vi.fn());

vi.mock("#/hooks/use-device-flow", () => ({
  useDeviceFlow: useDeviceFlowMock,
}));

vi.mock("react-i18next", () => ({
  useTranslation: (namespace?: unknown) => {
    useTranslationMock(namespace);
    return {
      t: (key: string) =>
        ({
          BACKEND$LOGIN_WITH_OPENHANDS: "Login with OpenHands Cloud",
          BACKEND$AUTH_STARTING: "Starting authorization",
          BACKEND$AUTH_AWAITING: "Waiting for authorization",
          BACKEND$AUTH_BROWSER_OPENED: "Complete login in your browser",
          BACKEND$AUTH_OPEN_MANUALLY: "Open the authorization page",
          BACKEND$AUTH_CANCEL: "Cancel",
          BACKEND$AUTH_RETRY: "Retry",
        })[key] ?? key,
    };
  },
}));

type DeviceFlowAuthProps = React.ComponentProps<typeof DeviceFlowAuth>;

function createFlow(
  overrides: Partial<UseDeviceFlowReturn> = {},
): UseDeviceFlowReturn {
  return {
    status: "idle",
    verificationUrl: null,
    userCode: null,
    apiKey: null,
    error: null,
    errorCode: null,
    start: vi.fn(),
    cancel: vi.fn(),
    reset: vi.fn(),
    ...overrides,
  };
}

function createProps(
  overrides: Partial<DeviceFlowAuthProps> = {},
): DeviceFlowAuthProps {
  return {
    host: "cloud.example.com",
    onSuccess: vi.fn(),
    testIdRoot: "cloud-login",
    ...overrides,
  };
}

function arrangeFlow(flow: UseDeviceFlowReturn) {
  useDeviceFlowMock.mockReset();
  useDeviceFlowMock.mockReturnValue(flow);
}

function arrangeFlowTransition(
  initial: UseDeviceFlowReturn,
  next: UseDeviceFlowReturn,
) {
  useDeviceFlowMock.mockReset();
  useDeviceFlowMock.mockReturnValueOnce(initial).mockReturnValue(next);
}

function createPopup({
  closed = false,
  initialHref = "about:blank",
  throwOnNavigation = false,
}: {
  closed?: boolean;
  initialHref?: string;
  throwOnNavigation?: boolean;
} = {}) {
  const close = vi.fn();
  const location = throwOnNavigation
    ? Object.defineProperty({}, "href", {
        configurable: true,
        get: () => initialHref,
        set: () => {
          throw new Error("Cross-origin navigation");
        },
      })
    : { href: initialHref };
  const popup = {
    closed,
    close,
    location,
  } as unknown as Window;

  return { popup, close, location: location as { href: string } };
}

afterEach(() => {
  useDeviceFlowMock.mockReset();
  useTranslationMock.mockReset();
  vi.restoreAllMocks();
});

describe("device-flow login", () => {
  it("renders the default styled login label and respects its disabled state", async () => {
    const user = userEvent.setup();
    const flow = createFlow();
    arrangeFlow(flow);
    const open = vi.spyOn(window, "open").mockReturnValue(createPopup().popup);

    render(<DeviceFlowAuth {...createProps({ isDisabled: true })} />);

    expect(useTranslationMock).toHaveBeenNthCalledWith(1, "openhands");
    expect(useTranslationMock).toHaveBeenNthCalledWith(2, "openhands");
    expect(useTranslationMock).toHaveBeenCalledTimes(2);
    expect(screen.getByTestId("cloud-login-device-flow")).toHaveClass(
      "flex",
      "flex-col",
      "gap-3",
    );
    const button = screen.getByTestId("cloud-login-login-button");
    expect(button).toHaveAccessibleName("Login with OpenHands Cloud");
    expect(button).toHaveTextContent("Login with OpenHands Cloud");
    expect(button).not.toHaveAttribute("aria-label");
    expect(button).toHaveClass("bg-primary");
    expect(button).toBeDisabled();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();

    await user.click(button);
    expect(flow.start).not.toHaveBeenCalled();
    expect(open).not.toHaveBeenCalled();
  });

  it("gives custom content an accessible label in a styled button", () => {
    arrangeFlow(createFlow());

    render(
      <DeviceFlowAuth
        {...createProps({
          idleButtonLabel: "Connect cloud account",
          idleButtonContent: <span aria-hidden>☁</span>,
          buttonVariant: "tertiary",
          buttonClassName: "custom-button",
        })}
      />,
    );

    const button = screen.getByTestId("cloud-login-login-button");
    expect(button).toHaveAccessibleName("Connect cloud account");
    expect(button).toHaveTextContent("☁");
    expect(button).not.toHaveTextContent("Connect cloud account");
    expect(button).toHaveClass("custom-button");
  });

  it("starts normalized authentication from a custom unstyled control", async () => {
    const user = userEvent.setup();
    const flow = createFlow();
    arrangeFlow(flow);
    const { popup, close } = createPopup();
    const open = vi.spyOn(window, "open").mockReturnValue(popup);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});

    const { unmount } = render(
      <DeviceFlowAuth
        {...createProps({
          host: "  cloud.example.com///  ",
          idleButtonLabel: "Reconnect",
          idleButtonContent: <span aria-hidden>↗</span>,
          buttonVariant: "unstyled",
          className: "compact-login",
          buttonClassName: "icon-button",
          analyticsSource: "add_backend_modal",
        })}
      />,
    );

    const button = screen.getByTestId("cloud-login-login-button");
    expect(button).toHaveAccessibleName("Reconnect");
    expect(button).toHaveClass("icon-button");
    expect(screen.getByTestId("cloud-login-device-flow")).toHaveClass(
      "compact-login",
    );

    await user.click(button);
    expect(open).toHaveBeenCalledWith("about:blank", "_blank");
    expect(flow.start).toHaveBeenCalledWith(
      "https://cloud.example.com",
      "add_backend_modal",
    );
    expect(warn).not.toHaveBeenCalled();

    unmount();
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("supports a text-only unstyled trigger while modal status is idle", () => {
    arrangeFlow(createFlow());

    render(
      <DeviceFlowAuth
        {...createProps({
          idleButtonLabel: "Reconnect",
          buttonVariant: "unstyled",
          statusDisplay: "modal",
        })}
      />,
    );

    const button = screen.getByTestId("cloud-login-login-button");
    expect(button).toHaveTextContent("Reconnect");
    expect(button).not.toHaveAttribute("aria-label");
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it.each([
    ["a blank host", "   "],
    ["a host with a username", "https://user@example.com"],
    ["a host with a password", "https://:secret@example.com"],
  ])("does not start authentication for %s", async (_reason, host) => {
    const user = userEvent.setup();
    const flow = createFlow();
    arrangeFlow(flow);
    const open = vi.spyOn(window, "open").mockReturnValue(createPopup().popup);

    render(<DeviceFlowAuth {...createProps({ host })} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));

    expect(open).not.toHaveBeenCalled();
    expect(flow.start).not.toHaveBeenCalled();
  });

  it("only recognizes an explicit scheme at the start of the host", async () => {
    const user = userEvent.setup();
    const flow = createFlow();
    arrangeFlow(flow);
    vi.spyOn(window, "open").mockReturnValue(createPopup().popup);

    render(
      <DeviceFlowAuth
        {...createProps({ host: "prefixhttps://cloud.example.com" })}
      />,
    );
    await user.click(screen.getByTestId("cloud-login-login-button"));

    expect(flow.start).toHaveBeenCalledWith(
      "https://prefixhttps://cloud.example.com",
      undefined,
    );
  });

  it("renders idle description and footer only while the flow is idle", () => {
    arrangeFlow(createFlow());
    const { rerender } = render(
      <DeviceFlowAuth
        {...createProps({
          idleDescription: <p>idle-description</p>,
          idleFooter: <p>idle-footer</p>,
        })}
      />,
    );
    expect(screen.getByText("idle-description")).toBeInTheDocument();
    expect(screen.getByText("idle-footer")).toBeInTheDocument();

    arrangeFlow(createFlow({ status: "starting" }));
    rerender(
      <DeviceFlowAuth
        {...createProps({
          idleDescription: <p>idle-description</p>,
          idleFooter: <p>idle-footer</p>,
        })}
      />,
    );
    expect(screen.queryByText("idle-description")).not.toBeInTheDocument();
    expect(screen.queryByText("idle-footer")).not.toBeInTheDocument();
  });

  it("shows an accessible inline starting state", () => {
    arrangeFlow(createFlow({ status: "starting" }));

    render(<DeviceFlowAuth {...createProps({ buttonVariant: "unstyled" })} />);

    const status = screen.getByTestId("cloud-login-auth-starting");
    expect(status).toHaveAttribute("role", "status");
    expect(status).toHaveTextContent("Starting authorization");
    expect(status).toHaveAttribute("aria-live", "polite");
    expect(status.querySelector("svg")).toHaveAttribute("aria-hidden", "true");
    expect(
      screen.queryByTestId("cloud-login-login-button"),
    ).not.toBeInTheDocument();
  });

  it("navigates its popup to a safe verification URL and cancels cleanly", async () => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const verificationUrl =
      "https://cloud.example.com/device?user_code=ABCD-EFGH";
    const awaitingFlow = createFlow({
      status: "awaiting_authorization",
      verificationUrl,
      userCode: "ABCD-EFGH",
    });
    arrangeFlowTransition(idleFlow, awaitingFlow);
    const { popup, close, location } = createPopup();
    vi.spyOn(window, "open").mockReturnValue(popup);
    const props = createProps();

    const { rerender } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(location.href).toBe(verificationUrl);
    const manualLink = screen.getByRole("link", { name: verificationUrl });
    expect(manualLink).toHaveAttribute("href", verificationUrl);
    expect(manualLink).toHaveAttribute("target", "_blank");
    expect(manualLink).toHaveAttribute("rel", "noopener noreferrer");
    expect(screen.getByRole("status")).toHaveTextContent(
      "Waiting for authorization",
    );

    await user.click(screen.getByTestId("cloud-login-auth-cancel"));
    expect(awaitingFlow.cancel).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
  });

  it("keeps the manual link available when the popup was blocked", async () => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const verificationUrl = "https://cloud.example.com/device";
    const awaitingFlow = createFlow({
      status: "awaiting_authorization",
      verificationUrl,
    });
    arrangeFlowTransition(idleFlow, awaitingFlow);
    const open = vi.spyOn(window, "open").mockReturnValue(null);
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const props = createProps();

    const { rerender } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(open).toHaveBeenCalledTimes(1);
    expect(warn).toHaveBeenCalledWith(
      "Popup blocked - user will need to use manual link",
    );
    expect(screen.getByRole("link", { name: verificationUrl })).toHaveAttribute(
      "href",
      verificationUrl,
    );
  });

  it("does not try to navigate a popup the user already closed", async () => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const verificationUrl = "https://cloud.example.com/device";
    const awaitingFlow = createFlow({
      status: "awaiting_authorization",
      verificationUrl,
    });
    arrangeFlowTransition(idleFlow, awaitingFlow);
    const { popup, location } = createPopup({ closed: true });
    const open = vi.spyOn(window, "open").mockReturnValue(popup);
    const props = createProps();

    const { rerender } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(location.href).toBe("about:blank");
    expect(open).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("link", { name: verificationUrl })).toBeVisible();
  });

  it("does not navigate when a verification URL arrives outside the awaiting state", async () => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const startingFlow = createFlow({
      status: "starting",
      verificationUrl: "https://cloud.example.com/device",
    });
    arrangeFlowTransition(idleFlow, startingFlow);
    const { popup, location } = createPopup();
    const open = vi.spyOn(window, "open").mockReturnValue(popup);
    const props = createProps();

    const { rerender } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(location.href).toBe("about:blank");
    expect(open).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("cloud-login-auth-starting")).toBeVisible();
  });

  it("opens a fallback popup when the original cannot be navigated", async () => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const verificationUrl = "https://cloud.example.com/device";
    const awaitingFlow = createFlow({
      status: "awaiting_authorization",
      verificationUrl,
    });
    arrangeFlowTransition(idleFlow, awaitingFlow);
    const original = createPopup({ throwOnNavigation: true });
    const fallback = createPopup({ initialHref: verificationUrl });
    const open = vi
      .spyOn(window, "open")
      .mockReturnValueOnce(original.popup)
      .mockReturnValueOnce(fallback.popup);
    const props = createProps();

    const { rerender, unmount } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(open).toHaveBeenNthCalledWith(
      2,
      verificationUrl,
      "_blank",
      "noopener,noreferrer",
    );

    unmount();
    expect(fallback.close).toHaveBeenCalledTimes(1);
  });

  it.each([
    ["an insecure URL", "http://cloud.example.com/device"],
    ["a malformed URL", "not a URL"],
  ])("does not expose or navigate to %s", async (_reason, verificationUrl) => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const awaitingFlow = createFlow({
      status: "awaiting_authorization",
      verificationUrl,
    });
    arrangeFlowTransition(idleFlow, awaitingFlow);
    const { popup, location } = createPopup();
    const open = vi.spyOn(window, "open").mockReturnValue(popup);
    const error = vi.spyOn(console, "error").mockImplementation(() => {});
    const props = createProps();

    const { rerender } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(location.href).toBe("about:blank");
    expect(open).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(error).toHaveBeenCalledWith("Invalid verification URL protocol");
  });

  it("waits safely when authorization has not supplied a URL yet", () => {
    arrangeFlow(
      createFlow({
        status: "awaiting_authorization",
        verificationUrl: null,
      }),
    );

    render(<DeviceFlowAuth {...createProps()} />);

    expect(screen.getByTestId("cloud-login-auth-awaiting")).toBeVisible();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  it("delivers the API key, resets the flow, and closes the popup", async () => {
    const user = userEvent.setup();
    const idleFlow = createFlow();
    const successFlow = createFlow({
      status: "success",
      apiKey: "api-key-123",
    });
    arrangeFlowTransition(idleFlow, successFlow);
    const { popup, close } = createPopup();
    vi.spyOn(window, "open").mockReturnValue(popup);
    const props = createProps();

    const { rerender } = render(<DeviceFlowAuth {...props} />);
    await user.click(screen.getByTestId("cloud-login-login-button"));
    rerender(<DeviceFlowAuth {...props} />);

    expect(props.onSuccess).toHaveBeenCalledWith("api-key-123");
    expect(props.onSuccess).toHaveBeenCalledTimes(1);
    expect(successFlow.reset).toHaveBeenCalledTimes(1);
    expect(close).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
  });

  it("does not report a successful flow that has no API key", () => {
    const flow = createFlow({ status: "success", apiKey: null });
    arrangeFlow(flow);
    const props = createProps();

    render(<DeviceFlowAuth {...props} />);

    expect(props.onSuccess).not.toHaveBeenCalled();
    expect(flow.reset).not.toHaveBeenCalled();
    expect(
      screen.queryByTestId("cloud-login-login-button"),
    ).not.toBeInTheDocument();
    expect(screen.queryByRole("status")).not.toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("delivers an initially available API key without requiring a popup", () => {
    const flow = createFlow({ status: "success", apiKey: "api-key-123" });
    arrangeFlow(flow);
    const props = createProps();

    expect(() => render(<DeviceFlowAuth {...props} />)).not.toThrow();

    expect(props.onSuccess).toHaveBeenCalledWith("api-key-123");
    expect(flow.reset).toHaveBeenCalledTimes(1);
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("does not deliver an API key before authentication succeeds", () => {
    const flow = createFlow({
      status: "starting",
      apiKey: "premature-api-key",
    });
    arrangeFlow(flow);
    const props = createProps();

    render(<DeviceFlowAuth {...props} />);

    expect(props.onSuccess).not.toHaveBeenCalled();
    expect(flow.reset).not.toHaveBeenCalled();
  });

  it("shows an authorization error and retries an explicit host", async () => {
    const user = userEvent.setup();
    const flow = createFlow({
      status: "error",
      error: "Authorization expired",
      errorCode: "expired_token",
    });
    arrangeFlow(flow);
    const { popup } = createPopup();
    const open = vi.spyOn(window, "open").mockReturnValue(popup);

    render(
      <DeviceFlowAuth
        {...createProps({ host: "  HTTP://cloud.example.com///  " })}
      />,
    );

    expect(screen.getByTestId("cloud-login-auth-error")).toHaveTextContent(
      "Authorization expired",
    );
    await user.click(screen.getByTestId("cloud-login-auth-retry"));

    expect(open).toHaveBeenCalledWith("about:blank", "_blank");
    expect(flow.start).toHaveBeenCalledWith(
      "HTTP://cloud.example.com",
      undefined,
    );
  });

  it("renders progress in a modal that ignores backdrop clicks but cancels on Escape", () => {
    const flow = createFlow({ status: "starting" });
    arrangeFlow(flow);

    render(<DeviceFlowAuth {...createProps({ statusDisplay: "modal" })} />);

    const dialog = screen.getByRole("dialog", {
      name: "Login with OpenHands Cloud",
    });
    const modal = screen.getByTestId("cloud-login-auth-modal");
    expect(within(modal).getByRole("status")).toHaveTextContent(
      "Starting authorization",
    );
    expect(
      within(screen.getByTestId("cloud-login-device-flow")).queryByRole(
        "status",
      ),
    ).not.toBeInTheDocument();

    fireEvent.click(dialog.firstElementChild as HTMLElement);
    expect(flow.cancel).not.toHaveBeenCalled();

    const runtimeErrors: unknown[] = [];
    const preventRuntimeError = (event: ErrorEvent) => {
      event.preventDefault();
      runtimeErrors.push(event.error);
    };
    window.addEventListener("error", preventRuntimeError);
    try {
      fireEvent.keyDown(window, { key: "Escape" });
    } finally {
      window.removeEventListener("error", preventRuntimeError);
    }

    expect(runtimeErrors).toEqual([]);
    expect(flow.cancel).toHaveBeenCalledTimes(1);
  });
});
