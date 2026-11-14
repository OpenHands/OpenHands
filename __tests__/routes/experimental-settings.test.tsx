import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import ExperimentalSettingsScreen from "#/routes/experimental-settings";
import { HttpError } from "@openhands/typescript-client";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: { id: "local-1", kind: "local" },
    orgId: null,
  }),
}));

const reactions = (enabled: boolean) => ({
  key: "conversation_reactions",
  label: "Conversation reactions",
  description: "The agent can put an emoji reaction on your messages.",
  enabled,
});

describe("Features settings", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("lists the server's experimental features and turns one on", async () => {
    // The server's list before and after the switch (read again to reconcile).
    vi.spyOn(experimentalFeaturesService, "listExperimentalFeatures")
      .mockResolvedValueOnce([reactions(false)])
      .mockResolvedValue([reactions(true)]);
    const update = vi
      .spyOn(experimentalFeaturesService, "updateExperimentalFeature")
      .mockResolvedValue(reactions(true));
    const user = userEvent.setup();

    renderWithProviders(<ExperimentalSettingsScreen />);

    const toggle = await screen.findByTestId(
      "experimental-switch-conversation_reactions",
    );
    expect(screen.getByText("Conversation reactions")).toBeInTheDocument();
    expect(
      screen.getByText("The agent can put an emoji reaction on your messages."),
    ).toBeInTheDocument();
    expect(toggle).not.toBeChecked();
    expect(screen.getByTestId("experimental-notice")).toHaveTextContent(
      "SETTINGS$EXPERIMENTAL_NOTICE",
    );

    await user.click(toggle);

    expect(update).toHaveBeenCalledWith(
      expect.objectContaining({
        backend: expect.objectContaining({ id: "local-1" }),
        key: "conversation_reactions",
        enabled: true,
      }),
      expect.anything(),
    );
    await waitFor(() => expect(toggle).toBeChecked());
  });

  it("says so when the server has no experimental features", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockResolvedValue([]);

    renderWithProviders(<ExperimentalSettingsScreen />);

    expect(
      await screen.findByText("SETTINGS$EXPERIMENTAL_EMPTY"),
    ).toBeInTheDocument();
  });

  it("treats an agent server without experimental features as having none", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockRejectedValue(new HttpError(404, "Not Found"));

    renderWithProviders(<ExperimentalSettingsScreen />);

    expect(
      await screen.findByText("SETTINGS$EXPERIMENTAL_EMPTY"),
    ).toBeInTheDocument();
  });

  it("shows an error, not an empty list, when the server fails", async () => {
    vi.spyOn(
      experimentalFeaturesService,
      "listExperimentalFeatures",
    ).mockRejectedValue(new HttpError(500, "Internal Server Error"));

    renderWithProviders(<ExperimentalSettingsScreen />);

    expect(
      await screen.findByText("SETTINGS$EXPERIMENTAL_LOAD_FAILED", undefined, {
        timeout: 10000,
      }),
    ).toBeInTheDocument();
  });
});
