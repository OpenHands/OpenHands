import React from "react";
import { renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { SecretsService } from "#/api/secrets-service";
import { useSaveFieldsAsSecrets } from "#/hooks/mutation/use-save-fields-as-secrets";
import type { MarketplaceField } from "@openhands/extensions/integrations";
import {
  displayErrorToast,
  displaySuccessToast,
} from "#/utils/custom-toast-handlers";

// Replace both toast functions with vi.fn() instances so we can assert on them.
vi.mock("#/utils/custom-toast-handlers", () => ({
  displaySuccessToast: vi.fn(),
  displayErrorToast: vi.fn(),
}));

// Minimal field factory — the hook only needs `key` and `label`.
const f = (key: string, label = key): MarketplaceField =>
  ({ key, label }) as unknown as MarketplaceField;

// The hook reads a QueryClient via useQueryClient(), so renderHook needs a
// provider. We hand back the client too so tests can spy on invalidateQueries.
let queryClient: QueryClient;
const createWrapper = () => {
  queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);
};

const renderSaveHook = () =>
  renderHook(() => useSaveFieldsAsSecrets(), { wrapper: createWrapper() });

describe("useSaveFieldsAsSecrets", () => {
  beforeEach(() => {
    vi.spyOn(SecretsService, "createSecret").mockResolvedValue();
    vi.mocked(displaySuccessToast).mockClear();
    vi.mocked(displayErrorToast).mockClear();
  });

  afterEach(() => vi.clearAllMocks());

  // ── early-return guard ──────────────────────────────────────────────────────

  it("does nothing when no fields are checked", () => {
    const { result } = renderSaveHook();
    result.current([f("KEY")], { KEY: "val" }, { KEY: false });
    expect(SecretsService.createSecret).not.toHaveBeenCalled();
  });

  it("skips a checked field whose value is an empty string", () => {
    const { result } = renderSaveHook();
    result.current([f("KEY")], { KEY: "" }, { KEY: true });
    expect(SecretsService.createSecret).not.toHaveBeenCalled();
  });

  it("skips a checked field whose value is whitespace-only", () => {
    const { result } = renderSaveHook();
    result.current([f("KEY")], { KEY: "   " }, { KEY: true });
    expect(SecretsService.createSecret).not.toHaveBeenCalled();
  });

  // ── createSecret call arguments ────────────────────────────────────────────

  it("calls createSecret with the trimmed value and field label as description", () => {
    const { result } = renderSaveHook();
    result.current(
      [f("API_KEY", "API Key")],
      { API_KEY: "  sk-secret  " },
      { API_KEY: true },
    );
    expect(SecretsService.createSecret).toHaveBeenCalledWith(
      "API_KEY",
      "sk-secret",
      "API Key",
    );
  });

  it("only calls createSecret for checked fields, skipping unchecked ones", () => {
    const { result } = renderSaveHook();
    result.current(
      [f("CHECKED"), f("SKIPPED")],
      { CHECKED: "val", SKIPPED: "val" },
      { CHECKED: true, SKIPPED: false },
    );
    expect(SecretsService.createSecret).toHaveBeenCalledTimes(1);
    expect(SecretsService.createSecret).toHaveBeenCalledWith(
      "CHECKED",
      "val",
      "CHECKED",
    );
  });

  // ── Promise.allSettled behaviour ────────────────────────────────────────────

  it("calls createSecret for every checked field even if one of them rejects", () => {
    vi.spyOn(SecretsService, "createSecret")
      .mockRejectedValueOnce(new Error("conflict"))
      .mockResolvedValue();
    const { result } = renderSaveHook();
    result.current(
      [f("A"), f("B")],
      { A: "val-a", B: "val-b" },
      { A: true, B: true },
    );
    // Both are called synchronously inside the allSettled map.
    expect(SecretsService.createSecret).toHaveBeenCalledTimes(2);
  });

  // ── toast behaviour ─────────────────────────────────────────────────────────

  it("shows only a success toast when all fields save successfully", async () => {
    const { result } = renderSaveHook();
    result.current([f("KEY")], { KEY: "val" }, { KEY: true });
    await waitFor(() =>
      expect(vi.mocked(displaySuccessToast)).toHaveBeenCalledTimes(1),
    );
    expect(vi.mocked(displayErrorToast)).not.toHaveBeenCalled();
  });

  it("shows only an error toast when all fields fail to save", async () => {
    vi.spyOn(SecretsService, "createSecret").mockRejectedValue(
      new Error("fail"),
    );
    const { result } = renderSaveHook();
    result.current([f("KEY")], { KEY: "val" }, { KEY: true });
    await waitFor(() =>
      expect(vi.mocked(displayErrorToast)).toHaveBeenCalledTimes(1),
    );
    expect(vi.mocked(displaySuccessToast)).not.toHaveBeenCalled();
  });

  it("shows both success and error toasts on partial failure", async () => {
    vi.spyOn(SecretsService, "createSecret")
      .mockResolvedValueOnce()
      .mockRejectedValueOnce(new Error("fail"));
    const { result } = renderSaveHook();
    result.current(
      [f("OK_KEY"), f("FAIL_KEY")],
      { OK_KEY: "val", FAIL_KEY: "val" },
      { OK_KEY: true, FAIL_KEY: true },
    );
    await waitFor(() => {
      expect(vi.mocked(displaySuccessToast)).toHaveBeenCalledTimes(1);
      expect(vi.mocked(displayErrorToast)).toHaveBeenCalledTimes(1);
    });
  });

  // ── cache invalidation ──────────────────────────────────────────────────────

  it("invalidates the secrets query caches after a successful save", async () => {
    const { result } = renderSaveHook();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    result.current([f("KEY")], { KEY: "val" }, { KEY: true });
    await waitFor(() => {
      expect(invalidateSpy).toHaveBeenCalledWith({
        queryKey: ["secrets-search"],
      });
      expect(invalidateSpy).toHaveBeenCalledWith({ queryKey: ["secrets"] });
    });
  });

  it("does not invalidate the secrets query caches when every save fails", async () => {
    vi.spyOn(SecretsService, "createSecret").mockRejectedValue(
      new Error("fail"),
    );
    const { result } = renderSaveHook();
    const invalidateSpy = vi.spyOn(queryClient, "invalidateQueries");
    result.current([f("KEY")], { KEY: "val" }, { KEY: true });
    await waitFor(() =>
      expect(vi.mocked(displayErrorToast)).toHaveBeenCalledTimes(1),
    );
    expect(invalidateSpy).not.toHaveBeenCalled();
  });
});
