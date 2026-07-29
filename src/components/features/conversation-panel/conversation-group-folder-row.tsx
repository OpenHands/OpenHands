import { motion } from "framer-motion";
import { Folder, FolderOpen, Plus } from "lucide-react";
import { useRef, type DragEvent, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { cn } from "#/utils/utils";
import type {
  ConversationGroupLaunch,
  GroupFolderDropPosition,
} from "./conversation-panel-list-helpers";
import { getGroupConversationPreview } from "./conversation-panel-list-helpers";

interface ConversationGroup {
  id: string;
  label: string;
  conversations: AppConversation[];
  launch: ConversationGroupLaunch;
}

interface ConversationGroupFolderRowProps {
  group: ConversationGroup;
  expanded: boolean;
  previewExpanded: boolean;
  isDragging: boolean;
  dropIndicatorPosition: GroupFolderDropPosition | null;
  animateLayout: boolean;
  isCreatingConversationFlow: boolean;
  activeConversationId?: string | null;
  onToggleExpanded: () => void;
  onDragStart: () => void;
  onDragEnd: () => void;
  onDragOver: (event: DragEvent<HTMLElement>) => void;
  onDragLeave: () => void;
  onDrop: (event: DragEvent<HTMLElement>) => void;
  onTogglePreviewExpanded: () => void;
  onLaunchFromGroup: () => void;
  renderConversationCard: (conversation: AppConversation) => ReactNode;
}

export function ConversationGroupFolderRow({
  group,
  expanded,
  previewExpanded,
  isDragging,
  dropIndicatorPosition,
  animateLayout,
  isCreatingConversationFlow,
  activeConversationId,
  onToggleExpanded,
  onDragStart,
  onDragEnd,
  onDragOver,
  onDragLeave,
  onDrop,
  onTogglePreviewExpanded,
  onLaunchFromGroup,
  renderConversationCard,
}: ConversationGroupFolderRowProps) {
  const { t } = useTranslation("openhands");
  const sectionRef = useRef<HTMLElement>(null);
  const headingId = `thread-folder-${group.id.replace(/[^a-zA-Z0-9_-]/g, "-")}`;
  const groupTestIdSuffix = group.id.replace(/[^a-zA-Z0-9_-]/g, "-");
  const { visibleConversations, isPreviewTruncated, isShowingAll } =
    getGroupConversationPreview(group.conversations, {
      expanded: previewExpanded,
      activeConversationId,
    });

  return (
    <motion.section
      ref={sectionRef}
      layout={animateLayout ? "position" : false}
      transition={{ type: "spring", stiffness: 600, damping: 45 }}
      aria-labelledby={headingId}
      data-testid={`thread-folder-${groupTestIdSuffix}`}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
      className="relative rounded-md"
    >
      {dropIndicatorPosition ? (
        <div
          aria-hidden
          data-testid={`thread-folder-drop-indicator-${groupTestIdSuffix}`}
          className={cn(
            "pointer-events-none absolute inset-x-0 z-10 h-0.5 rounded-full bg-[var(--oh-accent)]",
            dropIndicatorPosition === "before" ? "-top-0.5" : "-bottom-0.5",
          )}
        />
      ) : null}
      {/* Keep the row's footprint (height) while dragging but hide its
          content so the original slot reads as an empty placeholder. Use
          opacity (not visibility/display) because hiding the drag source
          synchronously during dragstart cancels the native drag in Chrome. */}
      <div className={cn(isDragging && "opacity-0")}>
        <div
          className={cn(
            "flex h-8 w-full min-w-0 items-center gap-0.5 rounded-md pl-2 pr-1 text-sm font-normal",
            "text-[var(--oh-muted)] transition-colors hover:bg-[var(--oh-surface-raised)] hover:text-white",
          )}
        >
          <button
            type="button"
            draggable
            id={headingId}
            aria-expanded={expanded}
            aria-controls={`thread-folder-content-${groupTestIdSuffix}`}
            data-testid={`thread-folder-drag-${groupTestIdSuffix}`}
            aria-label={
              expanded
                ? t(I18nKey.CONVERSATION_PANEL$COLLAPSE_FOLDER, {
                    label: group.label,
                  })
                : t(I18nKey.CONVERSATION_PANEL$EXPAND_FOLDER, {
                    label: group.label,
                  })
            }
            onClick={onToggleExpanded}
            onDragStart={(event) => {
              event.stopPropagation();
              const { dataTransfer } = event;
              if (dataTransfer) {
                dataTransfer.effectAllowed = "move";
                dataTransfer.setData("text/plain", group.id);
                const node = sectionRef.current;
                if (node && typeof dataTransfer.setDragImage === "function") {
                  const rect = node.getBoundingClientRect();
                  // Render the floating preview from an off-screen clone so we
                  // can give it a rounded, surfaced background without altering
                  // the in-list row, and anchor it to the exact grab point.
                  const dragImage = node.cloneNode(true) as HTMLElement;
                  dragImage.style.position = "fixed";
                  dragImage.style.top = "0";
                  dragImage.style.left = "-9999px";
                  dragImage.style.width = `${rect.width}px`;
                  dragImage.style.margin = "0";
                  dragImage.style.pointerEvents = "none";
                  dragImage.style.borderRadius = "0.5rem";
                  dragImage.style.padding = "0.25rem";
                  dragImage.style.backgroundColor = "var(--oh-surface-raised)";
                  dragImage.style.boxShadow = "0 8px 24px rgba(0, 0, 0, 0.35)";
                  document.body.appendChild(dragImage);
                  dataTransfer.setDragImage(
                    dragImage,
                    event.clientX - rect.left,
                    event.clientY - rect.top,
                  );
                  // Remove after the browser has rasterized the drag image.
                  window.setTimeout(() => dragImage.remove(), 0);
                }
              }
              onDragStart();
            }}
            onDragEnd={(event) => {
              event.stopPropagation();
              onDragEnd();
            }}
            className={cn(
              "group/folder flex min-h-8 min-w-0 flex-1 cursor-grab items-center gap-2 rounded-md py-1 text-left text-inherit outline-none active:cursor-grabbing",
              "focus-visible:ring-1 focus-visible:ring-[var(--oh-border)]",
            )}
          >
            <Folder
              className={cn(
                "h-4 w-4 shrink-0",
                expanded
                  ? "hidden group-hover/folder:block"
                  : "block group-hover/folder:hidden",
              )}
              aria-hidden
            />
            <FolderOpen
              className={cn(
                "h-4 w-4 shrink-0",
                expanded
                  ? "block group-hover/folder:hidden"
                  : "hidden group-hover/folder:block",
              )}
              aria-hidden
            />
            <span className="truncate">{group.label}</span>
          </button>
          <button
            type="button"
            className={cn(
              "inline-flex h-6 w-6 shrink-0 cursor-pointer items-center justify-center rounded-md",
              "text-inherit transition-colors",
              "hover:bg-white/10 hover:text-white",
              "focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-[var(--oh-border)]",
              "disabled:cursor-not-allowed disabled:opacity-50",
            )}
            disabled={isCreatingConversationFlow}
            aria-label={t(
              I18nKey.CONVERSATION_PANEL$ADD_CONVERSATION_TO_GROUP,
              {
                label: group.label,
              },
            )}
            data-testid={`add-conversation-to-group-${groupTestIdSuffix}`}
            onClick={(event) => {
              event.preventDefault();
              event.stopPropagation();
              onLaunchFromGroup();
            }}
          >
            <Plus
              className="h-3.5 w-3.5 shrink-0"
              aria-hidden
              strokeWidth={2}
            />
          </button>
        </div>
        {expanded ? (
          <div
            id={`thread-folder-content-${groupTestIdSuffix}`}
            className="mt-0.5 space-y-0.5"
          >
            {visibleConversations.map(renderConversationCard)}
            {isPreviewTruncated ? (
              <div className="pl-2">
                <button
                  type="button"
                  data-testid={`thread-folder-view-more-${groupTestIdSuffix}`}
                  onClick={onTogglePreviewExpanded}
                  className="cursor-pointer text-xs text-[var(--oh-text-dim)] hover:text-white"
                >
                  {isShowingAll
                    ? t(I18nKey.CONVERSATION_PANEL$LESS)
                    : t(I18nKey.CONVERSATION_PANEL$MORE)}
                </button>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </motion.section>
  );
}
