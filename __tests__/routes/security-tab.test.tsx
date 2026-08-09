import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SecurityTab from "#/routes/security-tab";

const sastMutateAsync = vi.fn();
const scaMutateAsync = vi.fn();

vi.mock("#/hooks/query/use-security-sast-scan", () => ({
  useSecuritySastScan: () => ({
    mutateAsync: sastMutateAsync,
    isPending: false,
    isError: false,
    error: null,
  }),
}));

vi.mock("#/hooks/query/use-security-sca-scan", () => ({
  useSecurityScaScan: () => ({
    mutateAsync: scaMutateAsync,
    isPending: false,
    isError: false,
    error: null,
  }),
}));

vi.mock("#/hooks/query/use-dependency-track-integration", () => ({
  useConversationDependencyTrackIntegration: () => ({
    isReady: true,
    isLoading: false,
  }),
}));

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string) => {
      const labels: Record<string, string> = {
        "COMMON$SECURITY": "Security",
        "SECURITY$SCAN_SAST": "Scan SAST",
        "SECURITY$SCAN_SCA": "Scan SCA",
        "SECURITY$SCANNING": "Scanning…",
        "SECURITY$SAST_SCAN_COMPLETE": "SAST scan complete",
        "SECURITY$SCA_SCAN_COMPLETE": "SCA scan complete",
        "SECURITY$SAST_TITLE": "SAST (OpenGrep)",
        "SECURITY$SCA_TITLE": "SCA (Syft + Dependency-Track)",
      };
      return labels[key] ?? key;
    },
  }),
}));

vi.mock("#/components/features/security/security-scan-results", () => ({
  SecurityScanResults: ({ result }: { result: unknown }) => (
    <div data-testid="security-sast-results-mock">
      {result ? "has-sast-result" : "no-sast-result"}
    </div>
  ),
}));

vi.mock("#/components/features/security/security-sca-results", () => ({
  SecurityScaResults: ({ result }: { result: unknown }) => (
    <div data-testid="security-sca-results-mock">
      {result ? "has-sca-result" : "no-sca-result"}
    </div>
  ),
}));

describe("SecurityTab", () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = new QueryClient({
      defaultOptions: { queries: { retry: false } },
    });
    sastMutateAsync.mockReset();
    scaMutateAsync.mockReset();
    sastMutateAsync.mockResolvedValue({
      tool: "opengrep",
      scannedAt: "2026-08-09T00:00:00.000Z",
      findings: [],
    });
    scaMutateAsync.mockResolvedValue({
      tool: "dependency-track",
      scannedAt: "2026-08-09T00:00:00.000Z",
      findings: [],
    });
  });

  it("renders SAST and SCA scan buttons and triggers scans", async () => {
    const user = userEvent.setup();

    render(
      <QueryClientProvider client={queryClient}>
        <SecurityTab />
      </QueryClientProvider>,
    );

    expect(screen.getByTestId("security-tab")).toBeInTheDocument();
    expect(screen.getByTestId("security-sast-scan-button")).toHaveTextContent(
      "Scan SAST",
    );
    expect(screen.getByTestId("security-sca-scan-button")).toHaveTextContent(
      "Scan SCA",
    );

    await user.click(screen.getByTestId("security-sast-scan-button"));

    await waitFor(() => {
      expect(sastMutateAsync).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByTestId("security-sast-results-mock")).toHaveTextContent(
        "has-sast-result",
      );
    });

    await user.click(screen.getByTestId("security-sca-scan-button"));

    await waitFor(() => {
      expect(scaMutateAsync).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByTestId("security-sca-results-mock")).toHaveTextContent(
        "has-sca-result",
      );
    });
  });
});
