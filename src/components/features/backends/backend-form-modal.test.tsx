import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import { BackendConnectionOptions } from "./backend-form-modal";

// Isolate the manual "add backend" column: the sibling Cloud-login column
// drives an OAuth device flow that is irrelevant to backend-kind selection.
vi.mock("./device-flow-auth", () => ({
  DeviceFlowAuth: () => null,
}));

afterEach(() => {
  vi.clearAllMocks();
});

describe("BackendConnectionOptions — manual backend type", () => {
  it("defaults the Type selector to Cloud for an OpenHands Cloud host", () => {
    render(
      <BackendConnectionOptions
        onConnected={vi.fn()}
        initialManualBackend={{ host: "https://app.all-hands.dev" }}
      />,
    );

    expect(screen.getByTestId("add-backend-kind-option-cloud")).toHaveAttribute(
      "aria-checked",
      "true",
    );
    expect(screen.getByTestId("add-backend-kind-option-local")).toHaveAttribute(
      "aria-checked",
      "false",
    );
  });

  it("defaults the Type selector to Local for a self-hosted host on a custom domain", () => {
    // A self-hosted OHE on a custom domain is indistinguishable from a local
    // agent-server by host alone, so it must not be auto-classified as Cloud.
    render(
      <BackendConnectionOptions
        onConnected={vi.fn()}
        initialManualBackend={{ host: "https://app.adorable.build.one" }}
      />,
    );

    expect(screen.getByTestId("add-backend-kind-option-local")).toHaveAttribute(
      "aria-checked",
      "true",
    );
  });

  it("persists the user-selected Cloud kind when adding a self-hosted OHE on a custom domain", async () => {
    const onConnected = vi.fn();
    render(
      <BackendConnectionOptions
        onConnected={onConnected}
        initialManualBackend={{
          name: "Adorable Enterprise",
          host: "app.adorable.build.one",
          apiKey: "oh-cloud-key",
        }}
      />,
    );

    // The user overrides the (Local) default to declare this custom domain a
    // Cloud app-server, then connects.
    fireEvent.click(screen.getByTestId("add-backend-kind-option-cloud"));
    fireEvent.click(screen.getByTestId("add-backend-submit"));

    await waitFor(() => expect(onConnected).toHaveBeenCalledTimes(1));
    expect(onConnected).toHaveBeenCalledWith(
      expect.objectContaining({
        kind: "cloud",
        name: "Adorable Enterprise",
        host: "https://app.adorable.build.one",
        apiKey: "oh-cloud-key",
      }),
      "manual",
      { agentServerVersion: null },
    );
  });
});
