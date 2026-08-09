import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { AxiosError } from "axios";
import { describe, expect, it, vi, beforeEach } from "vitest";
import McpService from "#/api/mcp-service/mcp-service.api";
import { OauthCallbackFallback } from "#/components/features/mcp-page/oauth-callback-fallback";

const VALID_URL = "http://localhost:55607/callback?code=abc123";

function renderExpanded() {
  render(<OauthCallbackFallback jobId="job-1" />);
  fireEvent.click(screen.getByTestId("mcp-oauth-callback-fallback-toggle"));
}

describe("OauthCallbackFallback", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("rejects a non-loopback URL without calling the service", async () => {
    // Arrange
    const submitSpy = vi.spyOn(McpService, "submitOAuthCallback");
    renderExpanded();

    // Act
    fireEvent.change(screen.getByTestId("mcp-oauth-callback-url"), {
      target: { value: "https://evil.example.com/callback?code=abc123" },
    });
    fireEvent.click(screen.getByTestId("mcp-oauth-callback-submit"));

    // Assert
    expect(
      await screen.findByTestId("mcp-oauth-callback-url-error"),
    ).toBeInTheDocument();
    expect(submitSpy).not.toHaveBeenCalled();
  });

  it("relays a valid URL and locks the form", async () => {
    // Arrange
    const submitSpy = vi
      .spyOn(McpService, "submitOAuthCallback")
      .mockResolvedValue({ ok: true, status: "succeeded", job_id: "job-1" });
    renderExpanded();

    // Act
    fireEvent.change(screen.getByTestId("mcp-oauth-callback-url"), {
      target: { value: VALID_URL },
    });
    fireEvent.click(screen.getByTestId("mcp-oauth-callback-submit"));

    // Assert
    await waitFor(() =>
      expect(submitSpy).toHaveBeenCalledWith("job-1", VALID_URL),
    );
    await waitFor(() =>
      expect(screen.getByTestId("mcp-oauth-callback-submit")).toBeDisabled(),
    );
  });

  it("relays a provider denial so the backend can fail the job immediately", async () => {
    // Arrange: a denial carries `error`, never `code` — the client must not
    // second-guess the backend and reject it.
    const denialUrl = "http://127.0.0.1:55607/cb?error=access_denied&state=xyz";
    const submitSpy = vi
      .spyOn(McpService, "submitOAuthCallback")
      .mockResolvedValue({ ok: false, status: "failed", job_id: "job-1" });
    renderExpanded();

    // Act
    fireEvent.change(screen.getByTestId("mcp-oauth-callback-url"), {
      target: { value: denialUrl },
    });
    fireEvent.click(screen.getByTestId("mcp-oauth-callback-submit"));

    // Assert
    await waitFor(() =>
      expect(submitSpy).toHaveBeenCalledWith("job-1", denialUrl),
    );
  });

  it("surfaces a backend rejection inline", async () => {
    // Arrange
    const axiosError = new AxiosError("Request failed");
    axiosError.response = {
      data: { detail: "Unexpected OAuth callback URL" },
      status: 400,
      statusText: "Bad Request",
      headers: {},
      config: { headers: {} },
    } as AxiosError["response"];
    vi.spyOn(McpService, "submitOAuthCallback").mockRejectedValue(axiosError);
    renderExpanded();

    // Act
    fireEvent.change(screen.getByTestId("mcp-oauth-callback-url"), {
      target: { value: VALID_URL },
    });
    fireEvent.click(screen.getByTestId("mcp-oauth-callback-submit"));

    // Assert
    expect(
      await screen.findByText("Unexpected OAuth callback URL"),
    ).toBeInTheDocument();
  });
});
