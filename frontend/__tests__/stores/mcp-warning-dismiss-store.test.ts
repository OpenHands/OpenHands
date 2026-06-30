import { beforeEach, describe, expect, it } from "vitest";
import { useMcpWarningDismissStore } from "#/stores/mcp-warning-dismiss-store";

describe("useMcpWarningDismissStore", () => {
  beforeEach(() => {
    useMcpWarningDismissStore.setState({ dismissedKeys: [] });
  });

  it("tracks dismissals per conversation and server", () => {
    const { dismiss, isDismissed } = useMcpWarningDismissStore.getState();
    expect(isDismissed("conv-a", "jira")).toBe(false);
    dismiss("conv-a", "jira");
    expect(useMcpWarningDismissStore.getState().isDismissed("conv-a", "jira")).toBe(
      true,
    );
    expect(useMcpWarningDismissStore.getState().isDismissed("conv-b", "jira")).toBe(
      false,
    );
  });
});
