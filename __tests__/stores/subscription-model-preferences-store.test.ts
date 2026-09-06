import { beforeEach, describe, expect, it } from "vitest";
import {
  isSubscriptionModelEnabled,
  useSubscriptionModelPreferencesStore,
} from "#/stores/subscription-model-preferences-store";

const STORAGE_KEY = "subscription-model-preferences";

describe("subscription-model-preferences store", () => {
  beforeEach(() => {
    window.localStorage.clear();
    useSubscriptionModelPreferencesStore.setState({ disabledKeys: [] });
  });

  it("treats unknown keys as enabled", () => {
    expect(isSubscriptionModelEnabled([], "chatgpt:gpt-5.2")).toBe(true);
  });

  it("persists a disabled model and can turn it back on", () => {
    useSubscriptionModelPreferencesStore
      .getState()
      .setEnabled("cursor-cli:gpt-5.2", false);

    expect(
      useSubscriptionModelPreferencesStore.getState().disabledKeys,
    ).toEqual(["cursor-cli:gpt-5.2"]);
    expect(
      isSubscriptionModelEnabled(
        useSubscriptionModelPreferencesStore.getState().disabledKeys,
        "cursor-cli:gpt-5.2",
      ),
    ).toBe(false);

    const persisted = JSON.parse(
      window.localStorage.getItem(STORAGE_KEY) ?? "{}",
    );
    expect(persisted.state.disabledKeys).toEqual(["cursor-cli:gpt-5.2"]);

    useSubscriptionModelPreferencesStore
      .getState()
      .setEnabled("cursor-cli:gpt-5.2", true);
    expect(
      useSubscriptionModelPreferencesStore.getState().disabledKeys,
    ).toEqual([]);
  });
});
