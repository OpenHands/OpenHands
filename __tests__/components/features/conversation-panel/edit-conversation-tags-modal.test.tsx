import { describe, expect, it, vi } from "vitest";
import {
  Briefcase,
  CircleUserRound,
  CloudCog,
  Folder,
  FolderGit2,
  FolderKey,
  GitBranch,
  House,
  Layers,
  Tag,
  Waypoints,
  Wrench,
} from "lucide-react";
import { FaGithub } from "react-icons/fa6";
import SlackIcon from "#/icons/slack.svg?react";
import { getConversationTagIcon } from "#/components/features/conversation-panel/conversation-card/conversation-tag-icons";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { renderWithProviders } from "test-utils";
import { EditConversationTagsModal } from "#/components/features/conversation-panel/edit-conversation-tags-modal";
import { StartTaskStatusBadge } from "#/components/features/conversation-panel/start-task-card/start-task-status-badge";
import { I18nKey } from "#/i18n/declaration";

describe("getConversationTagIcon", () => {
  it("maps known keys to related icons", () => {
    expect(getConversationTagIcon("owner", "alice")).toBe(CircleUserRound);
    expect(getConversationTagIcon("env", "prod")).toBe(CloudCog);
    expect(getConversationTagIcon("repo_name", "org/repo")).toBe(FolderGit2);
    expect(getConversationTagIcon("selected_branch", "main")).toBe(GitBranch);
    expect(
      getConversationTagIcon("archiveworkspacepath", "/workspace/project"),
    ).toBe(Folder);
    expect(getConversationTagIcon("worktools", "browser")).toBe(Wrench);
    expect(
      getConversationTagIcon("Workwsid", "cad8995a71e54b12b156d1d153be062f"),
    ).toBe(FolderKey);
  });

  it("uses mode-specific icons for Appmode values", () => {
    expect(getConversationTagIcon("Appmode", "work")).toBe(Briefcase);
    expect(getConversationTagIcon("app_mode", "personal")).toBe(House);
    expect(getConversationTagIcon("mode", "custom")).toBe(Layers);
  });

  it("prefers source brand icons for origin and git_provider values", () => {
    expect(getConversationTagIcon("origin", "slack")).toBe(SlackIcon);
    expect(getConversationTagIcon("git_provider", "GitHub")).toBe(FaGithub);
  });

  it("falls back to the key icon for unknown sources", () => {
    expect(getConversationTagIcon("origin", "custom-bot")).toBe(Waypoints);
  });

  it("falls back to Tag for unrecognized keys", () => {
    expect(getConversationTagIcon("mystery", "value")).toBe(Tag);
  });
});

describe("EditConversationTagsModal", () => {
  it("renders the current user-facing tags as rows and hides reserved keys", () => {
    renderWithProviders(
      <EditConversationTagsModal
        tags={{
          origin: "slack",
          owner: "alice",
          acpserver: "claude-code",
          automationname: "Nightly Audit",
        }}
        onConfirm={vi.fn()}
        onCancel={vi.fn()}
      />,
    );

    expect(screen.getByTestId("edit-tag-row-origin")).toBeInTheDocument();
    expect(screen.getByTestId("edit-tag-row-owner")).toBeInTheDocument();
    // Reserved/internal keys are untouchable — no row for them. Automation
    // provenance stays out too, so a user can't edit (or spoof) it.
    expect(
      screen.queryByTestId("edit-tag-row-acpserver"),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByTestId("edit-tag-row-automationname"),
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

    expect(screen.queryByTestId("edit-tag-row-owner")).not.toBeInTheDocument();

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

    // Save does not paper over the rejected input by discarding it — the
    // error stands and the modal stays open until the key is fixed or cleared.
    await user.click(screen.getByTestId("confirm-button"));
    expect(onConfirm).not.toHaveBeenCalled();
    expect(screen.getByTestId("edit-tags-error")).toHaveTextContent(
      "CONVERSATION$TAG_KEY_INVALID",
    );

    await user.clear(screen.getByTestId("new-tag-key-input"));
    await user.clear(screen.getByTestId("new-tag-value-input"));
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

    // Automation provenance keys are reserved too: a user-added
    // ``automationname`` would flip ``isAutomationConversation``.
    await user.clear(screen.getByTestId("new-tag-key-input"));
    await user.clear(screen.getByTestId("new-tag-value-input"));
    await user.type(screen.getByTestId("new-tag-key-input"), "automationname");
    await user.type(screen.getByTestId("new-tag-value-input"), "x");
    await user.click(screen.getByTestId("add-tag-button"));
    expect(screen.getByTestId("edit-tags-error")).toHaveTextContent(
      "CONVERSATION$TAG_KEY_RESERVED",
    );

    // Value over the 256-char backend cap.
    await user.clear(screen.getByTestId("new-tag-key-input"));
    await user.type(screen.getByTestId("new-tag-key-input"), "owner");
    await user.type(screen.getByTestId("new-tag-value-input"), "v".repeat(257));
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

  it("lets a bare tag be removed (merge must not resurrect it)", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{ work: "", acpserver: "claude-code" }}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    // The bare tag is listed alongside valued tags…
    expect(screen.getByTestId("edit-tag-row-work")).toBeInTheDocument();

    await user.click(screen.getByTestId("remove-tag-work"));
    await user.click(screen.getByTestId("confirm-button"));

    // …and its removal sticks — only the reserved tag is preserved.
    expect(onConfirm).toHaveBeenCalledWith({ acpserver: "claude-code" });
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
    // Focus returns to the key box for the next tag, not the value box.
    expect(screen.getByTestId("new-tag-key-input")).toHaveFocus();

    // Enter in the key input (bare tag).
    await user.type(screen.getByTestId("new-tag-key-input"), "work{Enter}");
    expect(screen.getByTestId("edit-tag-row-work")).toBeInTheDocument();
  });

  it("commits a typed-but-unadded tag when Save is pressed", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{ acpserver: "claude-code" }}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    // Typing a tag and going straight for Save is the natural flow — Add is an
    // affordance for entering several, not a toll on entering one. Dropping
    // the pending row here loses work with no warning.
    await user.type(screen.getByTestId("new-tag-key-input"), "owner");
    await user.type(screen.getByTestId("new-tag-value-input"), "alice");
    await user.click(screen.getByTestId("confirm-button"));

    expect(onConfirm).toHaveBeenCalledWith({
      acpserver: "claude-code",
      owner: "alice",
    });
  });

  it("commits a typed-but-unadded bare tag when Save is pressed", async () => {
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
    await user.click(screen.getByTestId("confirm-button"));

    expect(onConfirm).toHaveBeenCalledWith({ work: "" });
  });

  it("blocks Save on an invalid pending tag instead of dropping it", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{}}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("new-tag-key-input"), "My Project");
    await user.click(screen.getByTestId("confirm-button"));

    expect(screen.getByTestId("edit-tags-error")).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();
    // The typed text survives so it can be corrected.
    expect(screen.getByTestId("new-tag-key-input")).toHaveValue("My Project");
  });

  it("blocks Save when a value was typed without a key", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{}}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.type(screen.getByTestId("new-tag-value-input"), "alice");
    await user.click(screen.getByTestId("confirm-button"));

    expect(screen.getByTestId("edit-tags-error")).toBeInTheDocument();
    expect(onConfirm).not.toHaveBeenCalled();
  });

  it("saves normally when both inputs are empty", async () => {
    const user = userEvent.setup();
    const onConfirm = vi.fn();
    renderWithProviders(
      <EditConversationTagsModal
        tags={{ origin: "slack" }}
        onConfirm={onConfirm}
        onCancel={vi.fn()}
      />,
    );

    await user.click(screen.getByTestId("confirm-button"));

    expect(onConfirm).toHaveBeenCalledWith({ origin: "slack" });
  });
});

describe("StartTaskStatusBadge", () => {
  // react-i18next is globally mocked so `t` echoes the key; asserting the key
  // proves the label is resolved through i18n rather than a raw enum transform.
  it.each([
    ["STARTING_CONVERSATION", I18nKey.CONVERSATION$STARTING_CONVERSATION],
    ["READY", I18nKey.CONVERSATION$READY],
    ["ERROR", I18nKey.COMMON$ERROR],
  ] as const)(
    "localizes the label for the %s status",
    (taskStatus, expectedKey) => {
      renderWithProviders(<StartTaskStatusBadge taskStatus={taskStatus} />);

      expect(screen.getByText(expectedKey)).toBeInTheDocument();
    },
  );

  it("renders nothing for the WORKING status", () => {
    const { container } = renderWithProviders(
      <StartTaskStatusBadge taskStatus="WORKING" />,
    );

    expect(container).toBeEmptyDOMElement();
  });
});
