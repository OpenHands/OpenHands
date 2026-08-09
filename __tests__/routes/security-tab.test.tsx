import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SecurityTab from "#/routes/security-tab";

const mutateAsync = vi.fn();

vi.mock("#/hooks/query/use-security-sast-scan", () => ({
  useSecuritySastScan: () => ({
    mutateAsync,
    isPending: false,
    isError: false,
    error: null,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const labels: Record<string, string> = {
        "COMMON$SECURITY": "Security",
        "SECURITY$SCAN": "Scan",
        "SECURITY$SCANNING": "Scanning…",
        "SECURITY$SCAN_COMPLETE": "Scan complete",
        "SECURITY$SAST_TITLE": "SAST (OpenGrep)",
        "SECURITY$EMPTY_STATE": "Run a scan to find security issues.",
      };
      return labels[key] ?? key;
    },
  }),
}));

vi.mock("#/components/features/security/security-scan-results", () => ({
  SecurityScanResults: ({ result }: { result: unknown }) => (
    <div data-testid="security-results-mock">
      {result ? "has-result" : "no-result"}
    </div>
  ),
}));

describe("SecurityTab", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    mutateAsync.mockReset();
    mutateAsync.mockResolvedValue({
      tool: "opengrep",
      scannedAt: "2026-08-09T00:00:00.000Z",
      findings: [],
    });
  });

  it("renders the scan button and triggers a SAST scan", async () => {
    const user = userEvent.setup();

    render(
      <QueryClientProvider client={queryClient}>
        <SecurityTab />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("security-tab")).toBeInTheDocument();
    expect(screen.getByTestId("security-scan-button")).toHaveTextContent("Scan");

    await user.click(screen.getByTestId("security-scan-button"));

    await waitFor(() => {
      expect(mutateAsync).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByTestId("security-results-mock")).toHaveTextContent(
        "has-result",
      );
    });
  });
});
