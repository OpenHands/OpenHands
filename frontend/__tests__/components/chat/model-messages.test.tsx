import React from "react";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { LlmProfileSummary } from "#/api/settings-service/profiles-service.api";
import { ModelMessages } from "#/components/features/chat/model-messages";
import { useModelStore } from "#/stores/model-store";

// Trans needs an i18next instance to render — without one, it emits nothing
// in the test env. Render a deterministic string so we can assert on it; the
// mocked `cmd` slot wraps `values.name` in a <strong> so MonoComponent's
// semantics are exercised even though the real component isn't mounted.
vi.mock("react-i18next", async (importOriginal) => {
  const actual = await importOriginal<typeof import("react-i18next")>();
  return {
    ...actual,
    useTranslation: () => ({
      t: (key: string) => key,
      i18n: { language: "en", exists: () => false },
    }),
    Trans: ({
      i18nKey,
      values,
      components,
    }: {
      i18nKey: string;
      values?: Record<string, string>;
      components?: Record<string, React.ReactElement>;
    }) => {
      const cmd = components?.cmd;
      return (
        <span>
          {`${i18nKey}:`}
          {cmd ? React.cloneElement(cmd, {}, values?.name ?? "") : values?.name}
        </span>
      );
    },
  };
});

const CONV = "conv-1";

const profile = (
  name: string,
  overrides: Partial<LlmProfileSummary> = {},
): LlmProfileSummary => ({
  name,
  model: "anthropic/claude-sonnet-4-6",
  base_url: null,
  api_key_set: true,
  ...overrides,
});

describe("<ModelMessages />", () => {
  beforeEach(() => {
    useModelStore.setState({ entriesByConversation: {} });
  });

  it("renders nothing when there are no entries", () => {
    const { container } = render(
      <ModelMessages conversationId={CONV} anchorEventId={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders the empty-profiles hint", () => {
    useModelStore.getState().show(CONV, null, []);
    render(<ModelMessages conversationId={CONV} anchorEventId={null} />);
    expect(screen.getByText("MODEL$NO_SAVED_PROFILES")).toBeInTheDocument();
    expect(screen.getByText("MODEL$NO_PROFILES_HINT")).toBeInTheDocument();
    expect(
      screen.queryByRole("button", { name: /got it/i }),
    ).not.toBeInTheDocument();
  });

  it("starts collapsed and reveals profile rows then row details on expansion", async () => {
    useModelStore
      .getState()
      .show(CONV, null, [profile("default"), profile("scratch")]);
    const user = userEvent.setup();
    render(<ModelMessages conversationId={CONV} anchorEventId={null} />);

    expect(screen.getByText("MODEL$AVAILABLE_PROFILES")).toBeInTheDocument();
    // Outer toggle is collapsed: profile rows are not in the document yet.
    expect(
      screen.queryByRole("button", { name: /default/i }),
    ).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /expand/i }));

    // Profile rows are now visible, but per-row details remain collapsed.
    expect(screen.queryByText(/api_key:/i)).toBeNull();

    await user.click(screen.getByRole("button", { name: /default/i }));
    expect(screen.getByText(/api_key:\s+set/i)).toBeInTheDocument();
  });

  it("only renders entries whose anchor matches the prop", () => {
    useModelStore.getState().show(CONV, null, [profile("default")]);
    useModelStore.getState().show(CONV, "evt-1", [profile("scratch")]);

    // Anchor null → only the first entry is visible.
    const { unmount } = render(
      <ModelMessages conversationId={CONV} anchorEventId={null} />,
    );
    expect(screen.getAllByTestId("model-messages")).toHaveLength(1);
    expect(screen.getByText("MODEL$AVAILABLE_PROFILES")).toBeInTheDocument();
    unmount();

    // Anchor "evt-1" → only the second entry is visible (different profile name).
    render(<ModelMessages conversationId={CONV} anchorEventId="evt-1" />);
    expect(screen.getByText("MODEL$AVAILABLE_PROFILES")).toBeInTheDocument();
  });

  it("does not render entries from other conversations", () => {
    useModelStore.getState().show("other-conv", null, [profile("default")]);
    const { container } = render(
      <ModelMessages conversationId={CONV} anchorEventId={null} />,
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders a switch entry with the profile name in a <strong>", () => {
    useModelStore.getState().recordSwitch(CONV, null, "gpt-5");
    render(<ModelMessages conversationId={CONV} anchorEventId={null} />);

    // Switch branch is taken: the SWITCHED_TO_PROFILE key is rendered, and
    // none of the list-branch keys are.
    expect(screen.getByText(/MODEL\$SWITCHED_TO_PROFILE/)).toBeInTheDocument();
    expect(screen.queryByText("MODEL$AVAILABLE_PROFILES")).toBeNull();
    expect(screen.queryByText("MODEL$NO_SAVED_PROFILES")).toBeNull();

    // The profile name renders inside MonoComponent's <strong> wrapper.
    const name = screen.getByText("gpt-5");
    expect(name.tagName).toBe("STRONG");
    expect(name.className).toContain("font-mono");
  });

  it("filters switch entries by anchor like list entries", () => {
    useModelStore.getState().recordSwitch(CONV, "evt-77", "gpt-5");
    useModelStore.getState().recordSwitch(CONV, null, "claude-sonnet");

    const { unmount } = render(
      <ModelMessages conversationId={CONV} anchorEventId={null} />,
    );
    expect(screen.getByText("claude-sonnet")).toBeInTheDocument();
    expect(screen.queryByText("gpt-5")).toBeNull();
    unmount();

    render(<ModelMessages conversationId={CONV} anchorEventId="evt-77" />);
    expect(screen.getByText("gpt-5")).toBeInTheDocument();
    expect(screen.queryByText("claude-sonnet")).toBeNull();
  });
});
