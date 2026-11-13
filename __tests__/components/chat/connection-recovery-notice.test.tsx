import { fireEvent, render, screen } from "@testing-library/react";
import { expect, it, vi } from "vitest";
import { ConnectionRecoveryNotice } from "#/components/features/chat/connection-recovery-notice";

it("offers manual retry only after sustained recovery and disappears when connected", () => {
  const reconnect = vi.fn();
  const view = render(
    <ConnectionRecoveryNotice state="reconnecting" onRetry={reconnect} />,
  );
  expect(screen.getByRole("status")).toBeVisible();
  expect(screen.queryByRole("button")).not.toBeInTheDocument();
  view.rerender(
    <ConnectionRecoveryNotice state="retryable" onRetry={reconnect} />,
  );
  fireEvent.click(screen.getByRole("button"));
  expect(reconnect).toHaveBeenCalledOnce();
  view.rerender(<ConnectionRecoveryNotice state="idle" onRetry={reconnect} />);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});
