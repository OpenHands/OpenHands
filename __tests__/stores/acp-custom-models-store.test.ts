import { beforeEach, describe, expect, it } from "vitest";
import { useAcpCustomModelsStore } from "#/stores/acp-custom-models-store";

const STORAGE_KEY = "acp-custom-models";
const PROFILE_ID = "11111111-1111-1111-1111-111111111111";

describe("acp-custom-models store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useAcpCustomModelsStore.setState({ customModelsByProfileId: {} });
  });

  it("adds a custom model at the front of the profile's list", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-b");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toEqual(["model-b", "model-a"]);
  });

  it("trims whitespace before storing", () => {
    useAcpCustomModelsStore
      .getState()
      .addCustomModel(PROFILE_ID, "  model-a  ");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toEqual(["model-a"]);
  });

  it("ignores an empty or whitespace-only model id", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "");
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "   ");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toBeUndefined();
  });

  it("does not duplicate a model already remembered for the profile", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toEqual(["model-a"]);
  });

  it("keeps custom models scoped per profile id", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");
    useAcpCustomModelsStore
      .getState()
      .addCustomModel("other-profile", "model-b");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toEqual(["model-a"]);
    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[
        "other-profile"
      ],
    ).toEqual(["model-b"]);
  });

  it("removes a custom model from the profile's list", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-b");
    useAcpCustomModelsStore.getState().removeCustomModel(PROFILE_ID, "model-a");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toEqual(["model-b"]);
  });

  it("no-ops removing a model that isn't remembered", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");
    useAcpCustomModelsStore
      .getState()
      .removeCustomModel(PROFILE_ID, "not-there");

    expect(
      useAcpCustomModelsStore.getState().customModelsByProfileId[PROFILE_ID],
    ).toEqual(["model-a"]);
  });

  it("prunes entries for profiles that no longer exist", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");
    useAcpCustomModelsStore
      .getState()
      .addCustomModel("other-profile", "model-b");

    useAcpCustomModelsStore.getState().pruneMissingProfiles([PROFILE_ID]);

    const { customModelsByProfileId } = useAcpCustomModelsStore.getState();
    expect(customModelsByProfileId[PROFILE_ID]).toEqual(["model-a"]);
    expect(customModelsByProfileId).not.toHaveProperty("other-profile");
  });

  it("persists remembered custom models to localStorage", () => {
    useAcpCustomModelsStore.getState().addCustomModel(PROFILE_ID, "model-a");

    const persisted = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.customModelsByProfileId[PROFILE_ID]).toEqual([
      "model-a",
    ]);
  });
});
