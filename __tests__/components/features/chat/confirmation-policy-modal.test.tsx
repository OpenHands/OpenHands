import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  ConfirmationPolicyModal,
  getConfirmationPolicyMode,
} from "#/components/features/chat/confirmation-policy-modal";
import {
  clearSessionConfirmationPolicies,
  getConfirmationPolicySessionScope,
  getSessionConfirmationPolicy,
} from "#/services/confirmation-policy-session";
import type { Backend } from "#/api/backend-registry/types";

const mocks = vi.hoisted(() => ({
  getConfirmationPolicy: vi.fn(),
  setConfirmationPolicy: vi.fn(),
  displayErrorToast: vi.fn(),
  displaySuccessToast: vi.fn(),
  backend: {
    id: "local-a",
    name: "Local A",
    host: "http://local-a.example",
    apiKey: "key-a",
    kind: "local" as const,
    connectionRevision: 2,
  },
}));

vi.mock("#/contexts/active-backend-context", () => ({
  useActiveBackend: () => ({ backend: mocks.backend, orgId: null }),
}));

vi.mock(
  "#/api/conversation-service/agent-server-conversation-service.api",
  () => ({
    default: {
      getConfirmationPolicy: mocks.getConfirmationPolicy,
      setConfirmationPolicy: mocks.setConfirmationPolicy,
    },
  }),
);

vi.mock("#/utils/custom-toast-handlers", () => ({
  displayErrorToast: mocks.displayErrorToast,
  displaySuccessToast: mocks.displaySuccessToast,
}));

vi.mock("react-i18next", async () => {
  const actual =
    await vi.importActual<typeof import("react-i18next")>("react-i18next");
  const definitions = await import("#/i18n/translation.json");
  const translations = definitions.default as Record<
    string,
    Record<string, string>
  >;
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string, options?: Record<string, unknown>) =>
        (translations[key]?.en ?? key).replace(
          /\{\{(\w+)\}\}/g,
          (placeholder, name: string) =>
            options?.[name] === undefined ? placeholder : String(options[name]),
        ),
    }),
  };
});

describe("ConfirmationPolicyModal", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    clearSessionConfirmationPolicies();
    mocks.backend = {
      id: "local-a",
      name: "Local A",
      host: "http://local-a.example",
      apiKey: "key-a",
      kind: "local",
      connectionRevision: 2,
    };
    mocks.getConfirmationPolicy.mockResolvedValue({ kind: "AlwaysConfirm" });
    mocks.setConfirmationPolicy.mockResolvedValue(undefined);
  });

  it("loads the live policy and applies the selected policy", async () => {
    const onClose = vi.fn();
    render(
      <ConfirmationPolicyModal
        conversationId="conversation-1"
        conversationUrl="http://runtime.example"
        sessionApiKey="runtime-key"
        onClose={onClose}
      />,
    );

    const confirmEveryAction = await screen.findByRole("button", {
      name: "Confirm every action",
    });
    expect(confirmEveryAction).toHaveAttribute("aria-pressed", "true");
    expect(mocks.getConfirmationPolicy).toHaveBeenCalledWith(
      "conversation-1",
      "http://runtime.example",
      "runtime-key",
    );

    fireEvent.click(
      screen.getByRole("button", {
        name: "Confirm high-risk actions only",
      }),
    );

    await waitFor(() =>
      expect(mocks.setConfirmationPolicy).toHaveBeenCalledWith(
        "conversation-1",
        {
          kind: "ConfirmRisky",
          threshold: "HIGH",
          confirm_unknown: true,
        },
        "http://runtime.example",
        "runtime-key",
      ),
    );
    expect(mocks.displaySuccessToast).toHaveBeenCalledWith(
      "Confirmation policy set to: Confirm high-risk actions only",
    );
    expect(
      getSessionConfirmationPolicy(
        getConfirmationPolicySessionScope(mocks.backend),
      ),
    ).toEqual({
      kind: "ConfirmRisky",
      threshold: "HIGH",
      confirm_unknown: true,
    });
    expect(onClose).toHaveBeenCalledOnce();
  });

  it.each([
    ["Always approve actions (no confirmation)", { kind: "NeverConfirm" }],
    ["Confirm every action", { kind: "AlwaysConfirm" }],
  ])("maps %s to the expected SDK policy", async (label, expectedPolicy) => {
    render(
      <ConfirmationPolicyModal
        conversationId="conversation-1"
        onClose={vi.fn()}
      />,
    );

    fireEvent.click(await screen.findByRole("button", { name: label }));

    await waitFor(() =>
      expect(mocks.setConfirmationPolicy).toHaveBeenCalledWith(
        "conversation-1",
        expectedPolicy,
        undefined,
        undefined,
      ),
    );
  });

  it("does not save a session preference when the live update fails", async () => {
    const onClose = vi.fn();
    mocks.setConfirmationPolicy.mockRejectedValue(new Error("Save failed"));
    render(
      <ConfirmationPolicyModal
        conversationId="conversation-1"
        onClose={onClose}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", {
        name: "Confirm high-risk actions only",
      }),
    );

    await waitFor(() =>
      expect(mocks.displayErrorToast).toHaveBeenCalledWith("Save failed"),
    );
    expect(
      getSessionConfirmationPolicy(
        getConfirmationPolicySessionScope(mocks.backend),
      ),
    ).toBeNull();
    expect(onClose).not.toHaveBeenCalled();
  });

  it("attributes an async save to the backend that invoked it", async () => {
    let resolveSave!: () => void;
    mocks.setConfirmationPolicy.mockReturnValue(
      new Promise<void>((resolve) => {
        resolveSave = resolve;
      }),
    );
    const invokingBackend = { ...mocks.backend } satisfies Backend;
    const otherBackend = {
      id: "local-b",
      name: "Local B",
      host: "http://local-b.example",
      apiKey: "key-b",
      kind: "local",
      connectionRevision: 0,
    } satisfies Backend;
    render(
      <ConfirmationPolicyModal
        conversationId="conversation-1"
        onClose={vi.fn()}
      />,
    );

    fireEvent.click(
      await screen.findByRole("button", {
        name: "Confirm high-risk actions only",
      }),
    );
    mocks.backend = otherBackend;
    resolveSave();

    await waitFor(() =>
      expect(
        getSessionConfirmationPolicy(
          getConfirmationPolicySessionScope(invokingBackend),
        ),
      ).toEqual({
        kind: "ConfirmRisky",
        threshold: "HIGH",
        confirm_unknown: true,
      }),
    );
    expect(
      getSessionConfirmationPolicy(
        getConfirmationPolicySessionScope(otherBackend),
      ),
    ).toBeNull();
  });

  it("reports a load failure and closes the modal", async () => {
    const onClose = vi.fn();
    mocks.getConfirmationPolicy.mockRejectedValue(new Error("Policy missing"));

    render(
      <ConfirmationPolicyModal
        conversationId="conversation-1"
        onClose={onClose}
      />,
    );

    await waitFor(() =>
      expect(mocks.displayErrorToast).toHaveBeenCalledWith("Policy missing"),
    );
    expect(onClose).toHaveBeenCalledOnce();
  });
});

describe("getConfirmationPolicyMode", () => {
  it("accepts current and generated policy discriminators", () => {
    expect(getConfirmationPolicyMode({ kind: "NeverConfirm" })).toBe(
      "always-approve",
    );
    expect(
      getConfirmationPolicyMode({
        kind: "openhands__sdk__security__confirmation_policy__ConfirmRisky-Output__1",
      }),
    ).toBe("confirm-risky");
    expect(getConfirmationPolicyMode({ type: "always" })).toBe(
      "always-confirm",
    );
  });
});
