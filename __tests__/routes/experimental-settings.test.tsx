import { afterEach, describe, expect, it, vi } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import ExperimentalSettingsScreen from "#/routes/experimental-settings";
import { HttpError } from "@openhands/typescript-client";
import { SettingsClient } from "@openhands/typescript-client/clients";
import { experimentalFeaturesService } from "#/api/experimental-features-service/experimental-features-service.api";

const activeBackend = vi.hoisted(() => ({
  id: "local-1",
  kind: "local" as "local" | "cloud",
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({
    backend: activeBackend,
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
    activeBackend.kind = "local";
    vi.restoreAllMocks();
  });

  it("lists the server's experimental features and turns one on", async () => {
    const anotherFeature = {
      key: "future_experiment",
      label: "A server-defined experiment",
      description: "A feature Canvas does not know about.",
      enabled: true,
    };
    // The server's list before and after the switch (read again to reconcile).
    const list = vi
      .spyOn(experimentalFeaturesService, "listExperimentalFeatures")
      .mockResolvedValueOnce([reactions(false), anotherFeature])
      .mockResolvedValue([reactions(true), anotherFeature]);
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
    expect(
      screen.getByRole("checkbox", { name: anotherFeature.label }),
    ).toBeChecked();
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
    expect(
      screen.getByRole("checkbox", { name: anotherFeature.label }),
    ).toBeChecked();

    list.mockResolvedValue([
      reactions(true),
      { ...anotherFeature, enabled: false },
    ]);
    update.mockResolvedValue({ ...anotherFeature, enabled: false });
    await user.click(
      screen.getByRole("checkbox", { name: anotherFeature.label }),
    );

    await waitFor(() =>
      expect(
        screen.getByRole("checkbox", { name: anotherFeature.label }),
      ).not.toBeChecked(),
    );
    expect(toggle).toBeChecked();
    expect(update).toHaveBeenLastCalledWith(
      expect.objectContaining({ key: anotherFeature.key, enabled: false }),
      expect.anything(),
    );
  });

  it("shows no Cloud features without contacting the local settings API", async () => {
    activeBackend.kind = "cloud";
    const list = vi.spyOn(SettingsClient.prototype, "listExperimentalFeatures");

    renderWithProviders(<ExperimentalSettingsScreen />);

    expect(
      await screen.findByText("SETTINGS$EXPERIMENTAL_EMPTY"),
    ).toBeInTheDocument();
    expect(list).not.toHaveBeenCalled();
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
