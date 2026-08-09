import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import SecurityTab from "#/routes/security-tab";
import { NavigationProvider } from "#/context/navigation-context";
import { useSecurityScanResultsStore } from "#/stores/security-scan-results-store";

const CONVERSATION_ID = "test-conversation-id";

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
    i18n: { language: "en" },
    t: (key: string) => {
      const labels: Record<string, string> = {
        "COMMON$SECURITY": "Security",
        "SECURITY$SCAN_SAST": "Scan SAST",
        "SECURITY$SCAN_SCA": "Scan SCA",
        "SECURITY$SCANNING": "Scanning…",
        "SECURITY$SAST_SCAN_COMPLETE": "SAST scan complete",
        "SECURITY$SCA_SCAN_COMPLETE": "SCA scan complete",
      };
      return labels[key] ?? key;
    },
  }),
}));

vi.mock("#/components/features/security/security-findings-panel", () => ({
  SecurityFindingsPanel: ({
    sastResult,
    scaResult,
  }: {
    sastResult: unknown;
    scaResult: unknown;
  }) => (
    <div data-testid="security-findings-panel-mock">
      {sastResult ? "has-sast" : "no-sast"}|{scaResult ? "has-sca" : "no-sca"}
    </div>
  ),
}));

function renderSecurityTab() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <NavigationProvider
        value={{
          currentPath: `/conversations/${CONVERSATION_ID}`,
          conversationId: CONVERSATION_ID,
          isNavigating: false,
          navigate: vi.fn(),
        }}
      >
        <SecurityTab />
      </NavigationProvider>
    </QueryClientProvider>,
  );
}

describe("SecurityTab", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSecurityScanResultsStore.setState({ resultsByConversationId: {} });
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

    renderSecurityTab();

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
      expect(
        screen.getByTestId("security-findings-panel-mock"),
      ).toHaveTextContent("has-sast|no-sca");
    });

    await user.click(screen.getByTestId("security-sca-scan-button"));

    await waitFor(() => {
      expect(scaMutateAsync).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(
        screen.getByTestId("security-findings-panel-mock"),
      ).toHaveTextContent("has-sast|has-sca");
    });
  });

  it("keeps scan results after the tab unmounts and remounts", async () => {
    const user = userEvent.setup();

    const { unmount } = renderSecurityTab();

    await user.click(screen.getByTestId("security-sast-scan-button"));
    await waitFor(() => {
      expect(
        screen.getByTestId("security-findings-panel-mock"),
      ).toHaveTextContent("has-sast|no-sca");
    });

    unmount();
    renderSecurityTab();

    expect(
      screen.getByTestId("security-findings-panel-mock"),
    ).toHaveTextContent("has-sast|no-sca");
  });
});
