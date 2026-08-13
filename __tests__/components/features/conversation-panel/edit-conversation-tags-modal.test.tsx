import { describe, it, expect, vi } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EditConversationTagsModal } from "#/components/features/conversation-panel/edit-conversation-tags-modal";

describe("EditConversationTagsModal", () => {
  it("renders the current user-facing tags as rows and hides reserved keys", () => {
    renderWithProviders(
      <EditConversationTagsModal
        tags={{
          origin: "slack",
          owner: "alice",
          acpserver: "claude-code",
        }}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("edit-tag-row-origin")).toBeInTheDocument();
    expect(screen.getByTestId("edit-tag-row-owner")).toBeInTheDocument();
    // Reserved/internal keys are untouchable — no row for them.
    expect(
      screen.queryByTestId("edit-tag-row-acpserver"),
    ).not.toBeInTheDocument();
  });

  it("adds a new tag and confirms with the merged map preserving reserved tags", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{ origin: "slack", acpserver: "claude-code" }}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("new-tag-key-input"), "owner");
    await user.type(screen.getByTestId("new-tag-value-input"), "alice");
    await user.click(screen.getByTestId("add-tag-button"));

    expect(screen.getByTestId("edit-tag-row-owner")).toBeInTheDocument();

    await user.click(screen.getByTestId("confirm-button"));

    expect(onConfirm).toHaveBeenCalledWith({
      // The internal tag survives the replace-all PATCH untouched.
      acpserver: "claude-code",
      origin: "slack",
      owner: "alice",
    });
  });

  it("removes a tag row and omits it from the confirmed map", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{ origin: "slack", owner: "alice" }}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("remove-tag-owner"));

    expect(
      screen.queryByTestId("edit-tag-row-owner"),
    ).not.toBeInTheDocument();

    await user.click(screen.getByTestId("confirm-button"));

    expect(onConfirm).toHaveBeenCalledWith({ origin: "slack" });
  });

  it("blocks keys that violate the backend rule (lowercase alphanumeric only)", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{}}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("new-tag-key-input"), "Bad Key!");
    await user.type(screen.getByTestId("new-tag-value-input"), "x");
    await user.click(screen.getByTestId("add-tag-button"));

    expect(screen.getByTestId("edit-tags-error")).toHaveTextContent(
      "CONVERSATION$TAG_KEY_INVALID",
    );
    expect(screen.queryByTestId("edit-tags-rows")).not.toBeInTheDocument();

    await user.click(screen.getByTestId("confirm-button"));
    expect(onConfirm).toHaveBeenCalledWith({});
  });

  it("blocks reserved keys and over-long values", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{}}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    // Reserved key: allowed by the regex but untouchable by the user.
    await user.type(screen.getByTestId("new-tag-key-input"), "acpserver");
    await user.type(screen.getByTestId("new-tag-value-input"), "x");
    await user.click(screen.getByTestId("add-tag-button"));
    expect(screen.getByTestId("edit-tags-error")).toHaveTextContent(
      "CONVERSATION$TAG_KEY_RESERVED",
    );

    // Value over the 256-char backend cap.
    await user.clear(screen.getByTestId("new-tag-key-input"));
    await user.type(screen.getByTestId("new-tag-key-input"), "owner");
    await user.type(
      screen.getByTestId("new-tag-value-input"),
      "v".repeat(257),
    );
    await user.click(screen.getByTestId("add-tag-button"));
    expect(screen.getByTestId("edit-tags-error")).toHaveTextContent(
      "CONVERSATION$TAG_VALUE_INVALID",
    );

    expect(screen.queryByTestId("edit-tags-rows")).not.toBeInTheDocument();
  });

  it("treats an empty value as a bare tag and renders just the key", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{}}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("new-tag-key-input"), "work");
    await user.click(screen.getByTestId("add-tag-button"));

    const row = screen.getByTestId("edit-tag-row-work");
    expect(row).toHaveTextContent("work");
    expect(row).not.toHaveTextContent("=");

    await user.click(screen.getByTestId("confirm-button"));
    expect(onConfirm).toHaveBeenCalledWith({ work: "" });
  });

  it("adds the tag when Enter is pressed in either input", async () => {
    const user = userEvent.setup();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{}}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    // Enter in the value input.
    await user.type(screen.getByTestId("new-tag-key-input"), "owner");
    await user.type(screen.getByTestId("new-tag-value-input"), "alice{Enter}");
    expect(screen.getByTestId("edit-tag-row-owner")).toBeInTheDocument();

    // Enter in the key input (bare tag).
    await user.type(screen.getByTestId("new-tag-key-input"), "work{Enter}");
    expect(screen.getByTestId("edit-tag-row-work")).toBeInTheDocument();
  });
});
