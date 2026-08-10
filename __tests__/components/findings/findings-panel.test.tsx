/**
 * @spec PROJETOSIN-188 — findings panel Vitest coverage
 */

import React from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { FindingFpModal } from "#/components/features/findings/finding-fp-modal";
import { FindingsPage } from "#/components/features/findings/findings-page";
import {
  FindingsFilters,
  EMPTY_FINDINGS_FILTERS,
} from "#/components/features/findings/findings-filters";
import type { Finding } from "#/api/pentest/findings-types";
import { NavigationProvider } from "#/context/navigation-context";

vi.mock("#/hooks/use-pentest-capabilities", () => ({
  useHasPentestCapability: vi.fn(),
  usePentestCapabilitiesQuery: vi.fn(() => ({
    data: { profile: "pentester", capabilities: [] },
  })),
}));

vi.mock("#/hooks/query/use-findings", async () => {
  const actual = await vi.importActual<
    typeof import("#/hooks/query/use-findings")
  >("#/hooks/query/use-findings");
  return {
    ...actual,
    useFindingsList: vi.fn(),
    useFindingsStats: vi.fn(),
    useFindingDetail: vi.fn(),
    useTriageFinding: vi.fn(),
  };
});

vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

import { useHasPentestCapability } from "#/hooks/use-pentest-capabilities";
import {
  useFindingDetail,
  useFindingsList,
  useFindingsStats,
  useTriageFinding,
} from "#/hooks/query/use-findings";

const FINDING: Finding = {
  id: "finding-1",
  engagement_id: "eng-1",
  source_tool: "nuclei",
  title: "SQL Injection",
  description: "desc",
  severity: "high",
  asset: "app.example.com",
  endpoint: "/api/search",
  evidence: null,
  status: "new",
  dedupe_hash: null,
  fp_reason: null,
  triaged_by: null,
  triaged_at: null,
  defectdojo_id: null,
  defectdojo_synced_at: null,
  cvss_score: null,
  cve_ids: null,
  tags: null,
  created_by: "user",
  created_at: "2026-08-01T00:00:00.000Z",
  updated_at: "2026-08-01T00:00:00.000Z",
};

function renderPage(
  ui: React.ReactElement,
  opts?: { navigate?: (to: string) => void },
) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <NavigationProvider
        value={{
          currentPath: "/findings",
          conversationId: null,
          isNavigating: false,
          navigate: opts?.navigate ?? vi.fn(),
        }}
      >
        {ui}
      </NavigationProvider>
    </QueryClientProvider>,
  );
}

describe("Findings panel", () => {
  beforeEach(() => {
    vi.mocked(useHasPentestCapability).mockImplementation(
      (cap) =>
        cap === "pentest.findings.view" || cap === "pentest.findings.triage",
    );
    vi.mocked(useFindingsList).mockReturnValue({
      data: {
        items: [FINDING],
        total: 1,
        page: 1,
        page_size: 20,
        next_page: null,
      },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never);
    vi.mocked(useFindingsStats).mockReturnValue({
      data: {
        by_severity: { high: 1 },
        by_status: { new: 1 },
        total: 1,
      },
      isLoading: false,
      isError: false,
      error: null,
    } as never);
    vi.mocked(useFindingDetail).mockReturnValue({
      data: FINDING,
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn(),
    } as never);
    vi.mocked(useTriageFinding).mockReturnValue({
      mutateAsync: vi
        .fn()
        .mockResolvedValue({ ...FINDING, status: "confirmed" }),
      isPending: false,
    } as never);
  });

  // @spec PROJETOSIN-188 — AC-188-6
  it("shows empty state when engagement_id is missing", () => {
    renderPage(
      <FindingsPage
        engagementId={null}
        page={1}
        filters={EMPTY_FINDINGS_FILTERS}
        newOnly={false}
        onFiltersChange={vi.fn()}
        onClearFilters={vi.fn()}
        onToggleNewOnly={vi.fn()}
        onPageChange={vi.fn()}
      />,
    );
    expect(
      screen.getByTestId("findings-empty-no-engagement"),
    ).toBeInTheDocument();
  });

  // @spec PROJETOSIN-188 — AC-188-2
  it("shows forbidden when view capability is missing", () => {
    vi.mocked(useHasPentestCapability).mockReturnValue(false);
    renderPage(
      <FindingsPage
        engagementId="eng-1"
        page={1}
        filters={EMPTY_FINDINGS_FILTERS}
        newOnly={false}
        onFiltersChange={vi.fn()}
        onClearFilters={vi.fn()}
        onToggleNewOnly={vi.fn()}
        onPageChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("findings-forbidden")).toBeInTheDocument();
    expect(screen.queryByTestId("findings-table")).not.toBeInTheDocument();
  });

  // @spec PROJETOSIN-188 — AC-188-5
  it("hides triage actions without triage capability", () => {
    vi.mocked(useHasPentestCapability).mockImplementation(
      (cap) => cap === "pentest.findings.view",
    );
    renderPage(
      <FindingsPage
        engagementId="eng-1"
        page={1}
        filters={EMPTY_FINDINGS_FILTERS}
        newOnly={false}
        onFiltersChange={vi.fn()}
        onClearFilters={vi.fn()}
        onToggleNewOnly={vi.fn()}
        onPageChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("findings-table")).toBeInTheDocument();
    expect(
      screen.queryByTestId("findings-row-actions"),
    ).not.toBeInTheDocument();
  });

  // @spec PROJETOSIN-188 — AC-188-3
  it("filters by severity via toolbar", () => {
    const onChange = vi.fn();
    render(
      <FindingsFilters
        value={EMPTY_FINDINGS_FILTERS}
        toolOptions={["nuclei"]}
        onChange={onChange}
        onClear={vi.fn()}
        hasActiveFilters={false}
      />,
    );
    fireEvent.click(screen.getByText("FINDINGS$SEVERITY_CRITICAL"));
    expect(onChange).toHaveBeenCalledWith(
      expect.objectContaining({ severities: ["critical"] }),
    );
  });

  // @spec PROJETOSIN-188 — AC-188-4
  it("requires FP reason before submit", async () => {
    const onSubmit = vi.fn();
    render(
      <FindingFpModal
        isOpen
        isPending={false}
        onCancel={vi.fn()}
        onSubmit={onSubmit}
      />,
    );
    expect(screen.getByTestId("finding-fp-submit")).toBeDisabled();
    fireEvent.change(screen.getByTestId("finding-fp-reason"), {
      target: { value: "  " },
    });
    fireEvent.blur(screen.getByTestId("finding-fp-reason"));
    fireEvent.click(screen.getByTestId("finding-fp-submit"));
    expect(onSubmit).not.toHaveBeenCalled();

    fireEvent.change(screen.getByTestId("finding-fp-reason"), {
      target: { value: "Dev environment only" },
    });
    await waitFor(() => {
      expect(screen.getByTestId("finding-fp-submit")).not.toBeDisabled();
    });
    fireEvent.click(screen.getByTestId("finding-fp-submit"));
    expect(onSubmit).toHaveBeenCalledWith("Dev environment only");
  });

  // @spec PROJETOSIN-188 — D-188-1
  it("keeps mobile row actions outside the detail button", () => {
    renderPage(
      <FindingsPage
        engagementId="eng-1"
        page={1}
        filters={EMPTY_FINDINGS_FILTERS}
        newOnly={false}
        onFiltersChange={vi.fn()}
        onClearFilters={vi.fn()}
        onToggleNewOnly={vi.fn()}
        onPageChange={vi.fn()}
      />,
    );

    const actions = screen.getAllByTestId("findings-row-actions");
    expect(actions.length).toBeGreaterThan(0);
    for (const actionRoot of actions) {
      expect(actionRoot.closest("button")).toBeNull();
    }
  });

  // @spec PROJETOSIN-188 — D-188-2
  it("restores focus to the trigger when the FP modal closes", async () => {
    const onCancel = vi.fn();
    function Harness({ open }: { open: boolean }) {
      return (
        <>
          <button type="button" data-testid="fp-focus-trigger" />

          <FindingFpModal
            isOpen={open}
            isPending={false}
            onCancel={onCancel}
            onSubmit={vi.fn()}
          />
        </>
      );
    }

    const { rerender } = render(<Harness open={false} />);
    const trigger = screen.getByTestId("fp-focus-trigger");
    trigger.focus();
    expect(trigger).toHaveFocus();

    rerender(<Harness open />);
    await waitFor(() => {
      expect(screen.getByTestId("finding-fp-modal")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByTestId("finding-fp-cancel"));
    expect(onCancel).toHaveBeenCalled();
    rerender(<Harness open={false} />);

    await waitFor(() => {
      expect(trigger).toHaveFocus();
    });
  });

  // @spec PROJETOSIN-188 — AC-188-1
  it("renders populated table when engagement and capability are present", () => {
    renderPage(
      <FindingsPage
        engagementId="eng-1"
        page={1}
        filters={EMPTY_FINDINGS_FILTERS}
        newOnly={false}
        onFiltersChange={vi.fn()}
        onClearFilters={vi.fn()}
        onToggleNewOnly={vi.fn()}
        onPageChange={vi.fn()}
      />,
    );
    expect(screen.getByTestId("findings-page")).toBeInTheDocument();
    expect(screen.getByTestId("findings-table")).toBeInTheDocument();
    expect(
      screen.getAllByTestId("findings-row-finding-1").length,
    ).toBeGreaterThan(0);
  });
});
