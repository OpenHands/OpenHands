import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import userEvent from "@testing-library/user-event";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createRoutesStub } from "react-router";
import DeviceVerify from "#/routes/device-verify";

const { useIsAuthedMock, ENABLE_PROJ_USER_JOURNEY_MOCK } = vi.hoisted(() => ({
  useIsAuthedMock: vi.fn(() => ({
    data: false as boolean | undefined,
    isLoading: false,
  })),
  ENABLE_PROJ_USER_JOURNEY_MOCK: vi.fn(() => true),
}));

vi.mock("#/hooks/query/use-is-authed", () => ({
  useIsAuthed: () => useIsAuthedMock(),
}));

vi.mock("posthog-js/react", () => ({
  usePostHog: () => ({
    capture: vi.fn(),
  }),
}));

vi.mock("#/utils/feature-flags", () => ({
  ENABLE_PROJ_USER_JOURNEY: () => ENABLE_PROJ_USER_JOURNEY_MOCK(),
}));

const RouterStub = createRoutesStub([
  {
    Component: DeviceVerify,
    path: "/device-verify",
  },
]);

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
      },
    },
  });

  function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  }

  return Wrapper;
};

describe("DeviceVerify", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.stubGlobal("close", vi.fn());
    // Mock fetch for API calls
    vi.stubGlobal(
      "fetch",
      vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        }),
      ),
    );
    // Enable feature flag by default
    ENABLE_PROJ_USER_JOURNEY_MOCK.mockReturnValue(true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  describe("Loading State", () => {
    it("should show loading spinner while checking authentication", async () => {
      useIsAuthedMock.mockReturnValue({
        data: undefined,
        isLoading: true,
      });

      render(<RouterStub initialEntries={["/device-verify"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        const spinner = document.querySelector(".animate-spin");
        expect(spinner).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$PROCESSING")).toBeInTheDocument();
    });
  });

  describe("Not Authenticated State", () => {
    it("should show authentication required message when not authenticated", async () => {
      useIsAuthedMock.mockReturnValue({
        data: false,
        isLoading: false,
      });

      render(<RouterStub initialEntries={["/device-verify"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByText("DEVICE$AUTH_REQUIRED")).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$SIGN_IN_PROMPT")).toBeInTheDocument();
    });
  });

  describe("Authenticated without User Code", () => {
    it("should show manual code entry form when authenticated but no code in URL", async () => {
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      render(<RouterStub initialEntries={["/device-verify"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(
          screen.getByText("DEVICE$AUTHORIZATION_TITLE"),
        ).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$ENTER_CODE_PROMPT")).toBeInTheDocument();
      expect(screen.getByLabelText("DEVICE$CODE_INPUT_LABEL")).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: "DEVICE$CONTINUE" }),
      ).toBeInTheDocument();
    });

    it("should submit manually entered code", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      render(<RouterStub initialEntries={["/device-verify"]} />, {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(screen.getByLabelText("DEVICE$CODE_INPUT_LABEL")).toBeInTheDocument();
      });

      const input = screen.getByLabelText("DEVICE$CODE_INPUT_LABEL");
      await user.type(input, "TESTCODE");

      const submitButton = screen.getByRole("button", {
        name: "DEVICE$CONTINUE",
      });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          "/oauth/device/verify-authenticated",
          expect.objectContaining({
            method: "POST",
            headers: {
              "Content-Type": "application/x-www-form-urlencoded",
            },
            body: "user_code=TESTCODE",
            credentials: "include",
          }),
        );
      });
    });
  });

  describe("Authenticated with User Code", () => {
    it("should show authorization confirmation when authenticated with code in URL", async () => {
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByText("DEVICE$AUTHORIZATION_REQUEST"),
        ).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$CODE_LABEL")).toBeInTheDocument();
      expect(screen.getByText("ABC-123")).toBeInTheDocument();
      expect(screen.getByText("DEVICE$SECURITY_NOTICE")).toBeInTheDocument();
      expect(screen.getByText("DEVICE$SECURITY_WARNING")).toBeInTheDocument();
      expect(screen.getByText("DEVICE$CONFIRM_PROMPT")).toBeInTheDocument();
    });

    it("should show cancel and authorize buttons", async () => {
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$CANCEL" }),
        ).toBeInTheDocument();
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });
    });

    it("should include the LoginCTA component when feature flag is enabled", async () => {
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(screen.getByTestId("login-cta")).toBeInTheDocument();
      });
    });

    it("should not include the LoginCTA and be center-aligned when feature flag is disabled", async () => {
      ENABLE_PROJ_USER_JOURNEY_MOCK.mockReturnValue(false);
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByText("DEVICE$AUTHORIZATION_REQUEST"),
        ).toBeInTheDocument();
      });

      // CTA should not be rendered
      expect(screen.queryByTestId("login-cta")).not.toBeInTheDocument();

      // Container should use max-w-md (centered layout) instead of max-w-4xl
      const container = document.querySelector(".max-w-md");
      expect(container).toBeInTheDocument();
      expect(document.querySelector(".max-w-4xl")).not.toBeInTheDocument();

      // Authorization card should have mx-auto for centering
      const authCard = container?.querySelector(".mx-auto");
      expect(authCard).toBeInTheDocument();
    });

    it("should call window.close when cancel button is clicked", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$CANCEL" }),
        ).toBeInTheDocument();
      });

      const cancelButton = screen.getByRole("button", { name: "DEVICE$CANCEL" });
      await user.click(cancelButton);

      expect(window.close).toHaveBeenCalled();
    });

    it("should submit device verification when authorize button is clicked", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(mockFetch).toHaveBeenCalledWith(
          "/oauth/device/verify-authenticated",
          expect.objectContaining({
            method: "POST",
            headers: {
              "Content-Type": "application/x-www-form-urlencoded",
            },
            body: "user_code=ABC-123",
            credentials: "include",
          }),
        );
      });
    });
  });

  describe("Processing State", () => {
    it("should show processing spinner during verification", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      // Make fetch hang to show processing state
      const mockFetch = vi.fn(() => new Promise(() => {}));
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        const spinner = document.querySelector(".animate-spin");
        expect(spinner).toBeInTheDocument();
        expect(screen.getByText("DEVICE$PROCESSING")).toBeInTheDocument();
      });
    });
  });

  describe("Success State", () => {
    it("should show success message after successful verification", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(screen.getByText("DEVICE$SUCCESS_TITLE")).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$SUCCESS_MESSAGE")).toBeInTheDocument();
      // Should show success icon (checkmark)
      const successIcon = document.querySelector(".text-green-600");
      expect(successIcon).toBeInTheDocument();
    });

    it("should not show try again button on success", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: true,
          json: () => Promise.resolve({}),
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(screen.getByText("DEVICE$SUCCESS_TITLE")).toBeInTheDocument();
      });

      expect(
        screen.queryByRole("button", { name: "DEVICE$TRY_AGAIN" }),
      ).not.toBeInTheDocument();
    });
  });

  describe("Error State", () => {
    it("should show error message when verification fails with non-ok response", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 400,
          json: () => Promise.resolve({ error: "invalid_code" }),
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=INVALID"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(screen.getByText("DEVICE$ERROR_TITLE")).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$ERROR_FAILED")).toBeInTheDocument();
      // Should show error icon (X)
      const errorIcon = document.querySelector(".text-red-600");
      expect(errorIcon).toBeInTheDocument();
    });

    it("should show error message when fetch throws an exception", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() => Promise.reject(new Error("Network error")));
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=ABC-123"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(screen.getByText("DEVICE$ERROR_TITLE")).toBeInTheDocument();
      });

      expect(screen.getByText("DEVICE$ERROR_OCCURRED")).toBeInTheDocument();
    });

    it("should show try again button on error", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 400,
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      render(
        <RouterStub initialEntries={["/device-verify?user_code=INVALID"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$TRY_AGAIN" }),
        ).toBeInTheDocument();
      });
    });

    it("should reload page when try again button is clicked", async () => {
      const user = userEvent.setup();
      useIsAuthedMock.mockReturnValue({
        data: true,
        isLoading: false,
      });

      const mockFetch = vi.fn(() =>
        Promise.resolve({
          ok: false,
          status: 400,
        }),
      );
      vi.stubGlobal("fetch", mockFetch);

      const reloadMock = vi.fn();
      vi.stubGlobal("location", { reload: reloadMock });

      render(
        <RouterStub initialEntries={["/device-verify?user_code=INVALID"]} />,
        {
          wrapper: createWrapper(),
        },
      );

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$AUTHORIZE" }),
        ).toBeInTheDocument();
      });

      const authorizeButton = screen.getByRole("button", {
        name: "DEVICE$AUTHORIZE",
      });
      await user.click(authorizeButton);

      await waitFor(() => {
        expect(
          screen.getByRole("button", { name: "DEVICE$TRY_AGAIN" }),
        ).toBeInTheDocument();
      });

      const tryAgainButton = screen.getByRole("button", {
        name: "DEVICE$TRY_AGAIN",
      });
      await user.click(tryAgainButton);

      expect(reloadMock).toHaveBeenCalled();
    });
  });
});
