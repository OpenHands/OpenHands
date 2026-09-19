import { act, renderHook } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import {
  CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS,
  CLOUD_SANDBOX_RESUME_RETRY_BASE_DELAY_MS,
  CLOUD_SANDBOX_RESUME_RETRY_MAX_DELAY_MS,
  useCloudSandboxResume,
} from "#/hooks/use-cloud-sandbox-resume";

const baseProps = {
  enabled: true,
  conversationId: "conversation-1",
  sandboxId: "sandbox-1",
  sandboxStatus: "PAUSED",
  dataUpdatedAt: 1,
};

afterEach(() => {
  vi.useRealTimers();
});

describe("useCloudSandboxResume", () => {
  it("retries a failed resume on a later poll without remounting", async () => {
    vi.useFakeTimers();
    const onResume = vi
      .fn<() => Promise<unknown>>()
      .mockRejectedValueOnce(new Error("temporary failure"))
      .mockResolvedValueOnce(undefined);
    const onError = vi.fn();
    const { rerender } = renderHook((props) => useCloudSandboxResume(props), {
      initialProps: { ...baseProps, onResume, onError },
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(onResume).toHaveBeenCalledTimes(1);
    expect(onError).toHaveBeenCalledTimes(1);

    act(() => {
      vi.advanceTimersByTime(CLOUD_SANDBOX_RESUME_RETRY_BASE_DELAY_MS);
      rerender({ ...baseProps, dataUpdatedAt: 2, onResume, onError });
    });
    await act(async () => {
      await Promise.resolve();
    });
    expect(onResume).toHaveBeenCalledTimes(2);

    // A successful POST must not be duplicated while polling still reports
    // PAUSED, because the status update can lag behind the resume response.
    act(() => {
      vi.advanceTimersByTime(CLOUD_SANDBOX_RESUME_RETRY_MAX_DELAY_MS);
      rerender({ ...baseProps, dataUpdatedAt: 3, onResume, onError });
    });
    expect(onResume).toHaveBeenCalledTimes(2);
  });

  it("allows a new resume after RUNNING transitions back to PAUSED", async () => {
    const onResume = vi
      .fn<() => Promise<unknown>>()
      .mockResolvedValue(undefined);
    const onError = vi.fn();
    const { rerender } = renderHook((props) => useCloudSandboxResume(props), {
      initialProps: { ...baseProps, onResume, onError },
    });

    await act(async () => {
      await Promise.resolve();
    });
    expect(onResume).toHaveBeenCalledTimes(1);

    rerender({
      ...baseProps,
      sandboxStatus: "RUNNING",
      dataUpdatedAt: 2,
      onResume,
      onError,
    });
    rerender({ ...baseProps, dataUpdatedAt: 3, onResume, onError });
    await act(async () => {
      await Promise.resolve();
    });

    expect(onResume).toHaveBeenCalledTimes(2);
  });

  it("stops retrying after the bounded attempt limit", async () => {
    vi.useFakeTimers();
    const onResume = vi
      .fn<() => Promise<unknown>>()
      .mockRejectedValue(new Error("persistent failure"));
    const onError = vi.fn();
    const { rerender } = renderHook((props) => useCloudSandboxResume(props), {
      initialProps: { ...baseProps, onResume, onError },
    });

    for (
      let attempt = 1;
      attempt < CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS;
      attempt += 1
    ) {
      await act(async () => {
        await Promise.resolve();
      });
      const retryDelay = Math.min(
        CLOUD_SANDBOX_RESUME_RETRY_MAX_DELAY_MS,
        CLOUD_SANDBOX_RESUME_RETRY_BASE_DELAY_MS * 2 ** (attempt - 1),
      );
      act(() => {
        vi.advanceTimersByTime(retryDelay);
        rerender({
          ...baseProps,
          dataUpdatedAt: attempt + 1,
          onResume,
          onError,
        });
      });
    }
    await act(async () => {
      await Promise.resolve();
    });

    expect(onResume).toHaveBeenCalledTimes(CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS);
    act(() => {
      vi.advanceTimersByTime(CLOUD_SANDBOX_RESUME_RETRY_MAX_DELAY_MS);
      rerender({
        ...baseProps,
        dataUpdatedAt: CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS + 1,
        onResume,
        onError,
      });
    });
    expect(onResume).toHaveBeenCalledTimes(CLOUD_SANDBOX_RESUME_MAX_ATTEMPTS);
  });
});
