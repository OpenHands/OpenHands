import { act, fireEvent, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ComponentProps, HTMLAttributes } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { ConversationGroupFolderRow } from "#/components/features/conversation-panel/conversation-group-folder-row";
import { I18nKey } from "#/i18n/declaration";

interface MockMotionSectionProps extends HTMLAttributes<HTMLElement> {
  layout?: boolean | string;
  transition?: {
    type?: string;
    stiffness?: number;
    damping?: number;
  };
}

vi.mock("framer-motion", async () => {
  const { forwardRef } = await import("react");
  const MotionSection = forwardRef<HTMLElement, MockMotionSectionProps>(
    ({ layout, transition, ...props }, ref) => (
      <section
        ref={ref}
        data-layout={String(layout)}
        data-transition-type={transition?.type}
        data-transition-stiffness={transition?.stiffness}
        data-transition-damping={transition?.damping}
        {...props}
      />
    ),
  );
  MotionSection.displayName = "MotionSection";

  return { motion: { section: MotionSection } };
});

const useTranslationMock = vi.hoisted(() => vi.fn());

vi.mock("react-i18next", () => ({
  useTranslation: (namespace?: unknown) => {
    useTranslationMock(namespace);
    return {
      t: (key: string, options?: { label?: string }) =>
        options?.label ? `${key}:${options.label}` : key,
    };
  },
}));

const createConversation = (
  id: string,
  overrides: Partial<AppConversation> = {},
): AppConversation => ({
  id,
  created_by_user_id: "user-1",
  selected_repository: null,
  selected_branch: null,
  git_provider: null,
  title: `Conversation ${id}`,
  trigger: null,
  pr_number: [],
  llm_model: null,
  metrics: null,
  created_at: "2026-01-01T00:00:00.000Z",
  updated_at: "2026-01-01T00:00:00.000Z",
  execution_status: null,
  conversation_url: null,
  session_api_key: null,
  sandbox_id: null,
  sub_conversation_ids: [],
  ...overrides,
});

const createConversations = (count: number): AppConversation[] =>
  Array.from({ length: count }, (_, index) =>
    createConversation(`conversation-${index + 1}`),
  );

type FolderRowProps = ComponentProps<typeof ConversationGroupFolderRow>;

interface RenderRowOptions {
  conversations?: AppConversation[];
  group?: Partial<FolderRowProps["group"]>;
  props?: Partial<Omit<FolderRowProps, "group">>;
}

const renderRow = ({
  conversations = createConversations(3),
  group: groupOverrides = {},
  props: propOverrides = {},
}: RenderRowOptions = {}) => {
  const callbacks = {
    onToggleExpanded: vi.fn(),
    onDragStart: vi.fn(),
    onDragEnd: vi.fn(),
    onDragOver: vi.fn(),
    onDragLeave: vi.fn(),
    onDrop: vi.fn(),
    onTogglePreviewExpanded: vi.fn(),
    onLaunchFromGroup: vi.fn(),
    renderConversationCard: (conversation: AppConversation) => (
      <article key={conversation.id} data-testid="conversation-card">
        {conversation.title}
      </article>
    ),
  };
  const parentEvents = {
    onClick: vi.fn(),
    onDragStart: vi.fn(),
    onDragEnd: vi.fn(),
  };
  const group: FolderRowProps["group"] = {
    id: "ws:/workspace/alpha",
    label: "Workspace Alpha",
    conversations,
    launch: { workingDir: "/workspace/alpha" },
    ...groupOverrides,
  };
  const props: FolderRowProps = {
    group,
    expanded: false,
    previewExpanded: false,
    isDragging: false,
    dropIndicatorPosition: null,
    animateLayout: false,
    isCreatingConversationFlow: false,
    activeConversationId: null,
    ...callbacks,
    ...propOverrides,
  };
  const rendered = render(
    <div {...parentEvents}>
      <ConversationGroupFolderRow {...props} />
    </div>,
  );

  return {
    ...rendered,
    callbacks,
    group,
    parentEvents,
    props,
    user: userEvent.setup(),
  };
};

afterEach(() => {
  document
    .querySelectorAll<HTMLElement>('body > section[style*="left: -9999px"]')
    .forEach((dragImage) => dragImage.remove());
  vi.useRealTimers();
});

describe("conversation group folder interactions", () => {
  it("renders a sanitized, accessible collapsed folder and toggles it", async () => {
    useTranslationMock.mockClear();
    const { callbacks, user } = renderRow();
    const section = screen.getByTestId("thread-folder-ws--workspace-alpha");
    const heading = screen.getByRole("button", {
      name: `${I18nKey.CONVERSATION_PANEL$EXPAND_FOLDER}:Workspace Alpha`,
    });
    const row = heading.parentElement;
    const [folder, folderOpen] = heading.querySelectorAll("svg");
    const add = screen.getByRole("button", {
      name: `${I18nKey.CONVERSATION_PANEL$ADD_CONVERSATION_TO_GROUP}:Workspace Alpha`,
    });

    expect(useTranslationMock).toHaveBeenCalledWith("openhands");
    expect(section).toHaveAttribute(
      "aria-labelledby",
      "thread-folder-ws--workspace-alpha",
    );
    expect(section).toHaveAttribute("data-layout", "false");
    expect(section).toHaveAttribute("data-transition-type", "spring");
    expect(section).toHaveAttribute("data-transition-stiffness", "600");
    expect(section).toHaveAttribute("data-transition-damping", "45");
    expect(heading).toHaveAttribute("id", "thread-folder-ws--workspace-alpha");
    expect(heading).toHaveAttribute("aria-expanded", "false");
    expect(heading).toHaveAttribute(
      "aria-controls",
      "thread-folder-content-ws--workspace-alpha",
    );
    expect(section).toHaveTextContent("Workspace Alpha");
    expect(row).toHaveClass("flex", "text-[var(--oh-muted)]");
    expect(heading).toHaveClass("cursor-grab", "focus-visible:ring-1");
    expect(folder).toHaveClass("h-4", "block", "group-hover/folder:hidden");
    expect(folderOpen).toHaveClass("h-4", "hidden", "group-hover/folder:block");
    expect(add).toHaveClass(
      "h-6",
      "text-inherit",
      "hover:bg-white/10",
      "focus-visible:outline-none",
      "disabled:cursor-not-allowed",
    );
    expect(screen.queryByTestId("conversation-card")).toBeNull();
    expect(
      screen.queryByTestId("thread-folder-drop-indicator-ws--workspace-alpha"),
    ).toBeNull();

    await user.click(heading);

    expect(callbacks.onToggleExpanded).toHaveBeenCalledOnce();
  });

  it("renders every conversation without a preview action when the group is small", () => {
    renderRow({
      props: { expanded: true, animateLayout: true, isDragging: true },
    });
    const heading = screen.getByRole("button", {
      name: `${I18nKey.CONVERSATION_PANEL$COLLAPSE_FOLDER}:Workspace Alpha`,
    });
    const section = screen.getByTestId("thread-folder-ws--workspace-alpha");
    const [folder, folderOpen] = heading.querySelectorAll("svg");

    expect(section).toHaveAttribute("data-layout", "position");
    expect(heading).toHaveAttribute("aria-expanded", "true");
    expect(folder).toHaveClass("hidden", "group-hover/folder:block");
    expect(folderOpen).toHaveClass("block", "group-hover/folder:hidden");
    expect(screen.getAllByTestId("conversation-card")).toHaveLength(3);
    expect(
      screen.queryByTestId("thread-folder-view-more-ws--workspace-alpha"),
    ).toBeNull();
    expect(heading.parentElement?.parentElement).toHaveClass("opacity-0");
  });

  it("shows the first five conversations and offers to reveal the rest", async () => {
    const conversations = createConversations(6);
    const { callbacks, user } = renderRow({
      conversations,
      props: { expanded: true },
    });

    const cards = screen.getAllByTestId("conversation-card");
    expect(cards).toHaveLength(5);
    expect(cards.map((card) => card.textContent)).toEqual(
      conversations.slice(0, 5).map((conversation) => conversation.title),
    );
    expect(screen.queryByText("Conversation conversation-6")).toBeNull();

    const more = screen.getByTestId(
      "thread-folder-view-more-ws--workspace-alpha",
    );
    expect(more).toHaveTextContent(I18nKey.CONVERSATION_PANEL$MORE);
    await user.click(more);
    expect(callbacks.onTogglePreviewExpanded).toHaveBeenCalledOnce();
  });

  it("shows all truncated conversations and offers to collapse the preview", async () => {
    const { callbacks, user } = renderRow({
      conversations: createConversations(6),
      props: { expanded: true, previewExpanded: true },
    });

    expect(screen.getAllByTestId("conversation-card")).toHaveLength(6);
    const less = screen.getByTestId(
      "thread-folder-view-more-ws--workspace-alpha",
    );
    expect(less).toHaveTextContent(I18nKey.CONVERSATION_PANEL$LESS);
    await user.click(less);
    expect(callbacks.onTogglePreviewExpanded).toHaveBeenCalledOnce();
  });

  it("keeps an active conversation visible beyond the collapsed preview", () => {
    renderRow({
      conversations: createConversations(6),
      props: {
        expanded: true,
        activeConversationId: "conversation-6",
      },
    });

    const cards = screen
      .getAllByTestId("conversation-card")
      .map((card) => card.textContent);
    expect(cards).toEqual([
      "Conversation conversation-1",
      "Conversation conversation-2",
      "Conversation conversation-3",
      "Conversation conversation-4",
      "Conversation conversation-6",
    ]);
  });

  it.each([
    { position: "before" as const, expectedClass: "-top-0.5" },
    { position: "after" as const, expectedClass: "-bottom-0.5" },
  ])("renders a $position drop indicator", ({ position, expectedClass }) => {
    renderRow({ props: { dropIndicatorPosition: position } });

    expect(
      screen.getByTestId("thread-folder-drop-indicator-ws--workspace-alpha"),
    ).toHaveClass("pointer-events-none", expectedClass);
  });

  it("forwards folder-level drag-over, drag-leave, and drop interactions", () => {
    const { callbacks } = renderRow();
    const section = screen.getByTestId("thread-folder-ws--workspace-alpha");
    const dataTransfer = { dropEffect: "move" };

    fireEvent.dragOver(section, { dataTransfer });
    fireEvent.dragLeave(section, { dataTransfer });
    fireEvent.drop(section, { dataTransfer });

    expect(callbacks.onDragOver).toHaveBeenCalledOnce();
    expect(callbacks.onDragLeave).toHaveBeenCalledOnce();
    expect(callbacks.onDrop).toHaveBeenCalledOnce();
  });

  it("starts and ends a drag without bubbling to the surrounding list", () => {
    const { callbacks, parentEvents } = renderRow();
    const handle = screen.getByTestId("thread-folder-drag-ws--workspace-alpha");

    fireEvent.dragStart(handle);
    fireEvent.dragEnd(handle);

    expect(callbacks.onDragStart).toHaveBeenCalledOnce();
    expect(callbacks.onDragEnd).toHaveBeenCalledOnce();
    expect(parentEvents.onDragStart).not.toHaveBeenCalled();
    expect(parentEvents.onDragEnd).not.toHaveBeenCalled();
  });

  it("writes the group id into a native drag payload even without a custom drag image", () => {
    const { callbacks, group } = renderRow();
    const handle = screen.getByTestId("thread-folder-drag-ws--workspace-alpha");
    const dataTransfer = {
      effectAllowed: "copy",
      setData: vi.fn(),
    };

    fireEvent.dragStart(handle, { dataTransfer });

    expect(dataTransfer.effectAllowed).toBe("move");
    expect(dataTransfer.setData).toHaveBeenCalledWith("text/plain", group.id);
    expect(callbacks.onDragStart).toHaveBeenCalledOnce();
  });

  it("creates an anchored, styled drag image and removes it after rasterization", () => {
    vi.useFakeTimers();
    const { callbacks, group } = renderRow();
    const section = screen.getByTestId("thread-folder-ws--workspace-alpha");
    vi.spyOn(section, "getBoundingClientRect").mockReturnValue({
      x: 10,
      y: 20,
      left: 10,
      top: 20,
      right: 330,
      bottom: 68,
      width: 320,
      height: 48,
      toJSON: () => ({}),
    });
    const dataTransfer = {
      effectAllowed: "copy",
      setData: vi.fn(),
      setDragImage: vi.fn(),
    };

    const dragStartEvent = new Event("dragstart", {
      bubbles: true,
      cancelable: true,
    });
    Object.defineProperties(dragStartEvent, {
      clientX: { value: 50 },
      clientY: { value: 80 },
      dataTransfer: { value: dataTransfer },
    });
    fireEvent(
      screen.getByTestId("thread-folder-drag-ws--workspace-alpha"),
      dragStartEvent,
    );

    expect(dataTransfer.effectAllowed).toBe("move");
    expect(dataTransfer.setData).toHaveBeenCalledWith("text/plain", group.id);
    expect(dataTransfer.setDragImage).toHaveBeenCalledOnce();
    const [dragImage, offsetX, offsetY] = dataTransfer.setDragImage.mock
      .calls[0] as [HTMLElement, number, number];
    expect(offsetX).toBe(40);
    expect(offsetY).toBe(60);
    expect(document.body).toContainElement(dragImage);
    expect(dragImage.style.position).toBe("fixed");
    expect(dragImage.style.top).toBe("0px");
    expect(dragImage.style.left).toBe("-9999px");
    expect(dragImage.style.width).toBe("320px");
    expect(dragImage.style.margin).toBe("0px");
    expect(dragImage.style.pointerEvents).toBe("none");
    expect(dragImage.style.borderRadius).toBe("0.5rem");
    expect(dragImage.style.padding).toBe("0.25rem");
    expect(dragImage.style.backgroundColor).toBe("var(--oh-surface-raised)");
    expect(dragImage.style.boxShadow).toBe("0 8px 24px rgba(0, 0, 0, 0.35)");
    expect(callbacks.onDragStart).toHaveBeenCalledOnce();

    act(() => vi.runAllTimers());
    expect(document.body).not.toContainElement(dragImage);
  });

  it("launches from the group without bubbling or performing a default action", () => {
    const { callbacks, parentEvents } = renderRow();
    const add = screen.getByTestId(
      "add-conversation-to-group-ws--workspace-alpha",
    );
    const event = new MouseEvent("click", {
      bubbles: true,
      cancelable: true,
    });

    act(() => add.dispatchEvent(event));

    expect(event.defaultPrevented).toBe(true);
    expect(callbacks.onLaunchFromGroup).toHaveBeenCalledOnce();
    expect(parentEvents.onClick).not.toHaveBeenCalled();
  });

  it("prevents another group launch while conversation creation is active", async () => {
    const { callbacks, user } = renderRow({
      props: { isCreatingConversationFlow: true },
    });
    const add = screen.getByTestId(
      "add-conversation-to-group-ws--workspace-alpha",
    );

    expect(add).toBeDisabled();
    await user.click(add);
    expect(callbacks.onLaunchFromGroup).not.toHaveBeenCalled();
  });

  it("renders conversation cards inside the labelled folder content region", () => {
    renderRow({ props: { expanded: true } });
    const heading = screen.getByTestId(
      "thread-folder-drag-ws--workspace-alpha",
    );
    const content = document.getElementById(
      heading.getAttribute("aria-controls") ?? "",
    );

    expect(content).not.toBeNull();
    expect(
      within(content as HTMLElement).getAllByTestId("conversation-card"),
    ).toHaveLength(3);
  });
});
