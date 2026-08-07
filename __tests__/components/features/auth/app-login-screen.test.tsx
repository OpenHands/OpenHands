import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import AppLoginScreen from "#/components/features/auth/app-login-screen";
import { AppLoginService } from "#/api/app-login-service";

vi.mock("#/api/app-login-service", () => ({
  AppLoginService: {
    login: vi.fn(),
  },
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => key,
  }),
}));

function renderScreen() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={client}>
      <AppLoginScreen />
    </QueryClientProvider>,
  );
}

describe("AppLoginScreen", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("submits username and password through AppLoginService", async () => {
    const user = userEvent.setup();
    vi.mocked(AppLoginService.login).mockResolvedValue({
      ok: true,
      username: "heimdallsec",
    });

    renderScreen();

    await user.type(screen.getByTestId("app-login-username"), "heimdallsec");
    await user.type(screen.getByTestId("app-login-password"), "heimdallsec");
    await user.click(screen.getByTestId("app-login-submit"));

    await waitFor(() => {
      expect(AppLoginService.login).toHaveBeenCalledWith(
        "heimdallsec",
        "heimdallsec",
      );
    });
  });

  it("shows an error when credentials are rejected", async () => {
    const user = userEvent.setup();
    vi.mocked(AppLoginService.login).mockResolvedValue({
      ok: false,
      error: "Invalid username or password",
    });

    renderScreen();

    await user.type(screen.getByTestId("app-login-username"), "heimdallsec");
    await user.type(screen.getByTestId("app-login-password"), "wrong");
    await user.click(screen.getByTestId("app-login-submit"));

    expect(await screen.findByTestId("app-login-error")).toBeInTheDocument();
  });
});
