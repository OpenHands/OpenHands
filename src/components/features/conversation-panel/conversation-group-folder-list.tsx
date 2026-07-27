import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type DragEvent,
  type ReactNode,
} from "react";
import { useTranslation } from "react-i18next";
import type { AppConversation } from "#/api/conversation-service/agent-server-conversation-service.types";
import { I18nKey } from "#/i18n/declaration";
import { ConversationGroupFolderRow } from "./conversation-group-folder-row";
import {
  moveGroupFolderOrder,
  type ConversationGroupLaunch,
  type GroupFolderDropPosition,
} from "./conversation-panel-list-helpers";

interface ConversationGroup {
  id: string;
  label: string;
  conversations: AppConversation[];
  launch: ConversationGroupLaunch;
}

interface ConversationGroupFolderListProps {
  groups: ConversationGroup[];
  groupIds: readonly string[];
  groupFolderOrder: readonly string[];
  setGroupFolderOrder: (order: readonly string[]) => void;
  collapsedGroupIds: ReadonlySet<string>;
  expandedGroupPreviewIds: ReadonlySet<string>;
  onToggleGroupCollapsed: (groupId: string) => void;
  onToggleGroupPreviewExpanded: (groupId: string) => void;
  isCreatingConversationFlow: boolean;
  activeConversationId?: string | null;
  onLaunchFromGroup: (launch: ConversationGroupLaunch) => void;
  renderConversationCard: (conversation: AppConversation) => ReactNode;
}

export function ConversationGroupFolderList({
  groups,
  groupIds,
  groupFolderOrder,
  setGroupFolderOrder,
  collapsedGroupIds,
  expandedGroupPreviewIds,
  onToggleGroupCollapsed,
  onToggleGroupPreviewExpanded,
  isCreatingConversationFlow,
  activeConversationId,
  onLaunchFromGroup,
  renderConversationCard,
}: ConversationGroupFolderListProps) {
  const { t } = useTranslation("openhands");
  const [draggedGroupId, setDraggedGroupId] = useState<string | null>(null);
  const [dropTargetGroupId, setDropTargetGroupId] = useState<string | null>(
    null,
  );
  const [dropPosition, setDropPosition] =
    useState<GroupFolderDropPosition | null>(null);
  // Layout animation is only enabled around a drag-and-drop reorder so that
  // expand/collapse clicks don't trigger sibling repositioning animations.
  const [animateLayout, setAnimateLayout] = useState(false);
  const animateLayoutTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  useEffect(
    () => () => {
      if (animateLayoutTimeoutRef.current) {
        clearTimeout(animateLayoutTimeoutRef.current);
      }
    },
    [],
  );

  const stopAnimatingLayoutSoon = useCallback(() => {
    if (animateLayoutTimeoutRef.current) {
      clearTimeout(animateLayoutTimeoutRef.current);
    }
    animateLayoutTimeoutRef.current = setTimeout(() => {
      setAnimateLayout(false);
      animateLayoutTimeoutRef.current = null;
    }, 300);
  }, []);

  const resetDragState = useCallback(() => {
    setDraggedGroupId(null);
    setDropTargetGroupId(null);
    setDropPosition(null);
  }, []);

  const computeDropPosition = useCallback(
    (event: DragEvent<HTMLElement>): GroupFolderDropPosition => {
      const rect = event.currentTarget.getBoundingClientRect();
      return event.clientY < rect.top + rect.height / 2 ? "before" : "after";
    },
    [],
  );

  const handleDrop = useCallback(
    (targetGroupId: string, position: GroupFolderDropPosition) => {
      if (!draggedGroupId || draggedGroupId === targetGroupId) {
        resetDragState();
        stopAnimatingLayoutSoon();
        return;
      }

      setGroupFolderOrder(
        moveGroupFolderOrder(
          groupFolderOrder,
          groupIds,
          draggedGroupId,
          targetGroupId,
          position,
        ),
      );
      resetDragState();
      // Keep layout animation active across the reorder render, then disable
      // it once the spring settles.
      stopAnimatingLayoutSoon();
    },
    [
      draggedGroupId,
      groupFolderOrder,
      groupIds,
      resetDragState,
      setGroupFolderOrder,
      stopAnimatingLayoutSoon,
    ],
  );

  return (
    <nav
      aria-label={t(I18nKey.SIDEBAR$CONVERSATIONS)}
      className="space-y-1 md:space-y-0.5 pb-1"
    >
      {groups.map((group) => (
        <ConversationGroupFolderRow
          key={group.id}
          group={group}
          expanded={!collapsedGroupIds.has(group.id)}
          previewExpanded={expandedGroupPreviewIds.has(group.id)}
          isDragging={draggedGroupId === group.id}
          dropIndicatorPosition={
            dropTargetGroupId === group.id && draggedGroupId !== group.id
              ? dropPosition
              : null
          }
          animateLayout={animateLayout}
          isCreatingConversationFlow={isCreatingConversationFlow}
          activeConversationId={activeConversationId}
          onToggleExpanded={() => onToggleGroupCollapsed(group.id)}
          onDragStart={() => {
            setAnimateLayout(true);
            setDraggedGroupId(group.id);
          }}
          onDragEnd={() => {
            resetDragState();
            stopAnimatingLayoutSoon();
          }}
          onDragOver={(event) => {
            event.preventDefault();
            const { dataTransfer } = event;
            if (dataTransfer) {
              dataTransfer.dropEffect = "move";
            }
            setDropTargetGroupId(group.id);
            setDropPosition(computeDropPosition(event));
          }}
          onDragLeave={() => {
            setDropTargetGroupId((current) =>
              current === group.id ? null : current,
            );
          }}
          onDrop={(event) => {
            event.preventDefault();
            handleDrop(group.id, computeDropPosition(event));
          }}
          onTogglePreviewExpanded={() => onToggleGroupPreviewExpanded(group.id)}
          onLaunchFromGroup={() => onLaunchFromGroup(group.launch)}
          renderConversationCard={renderConversationCard}
        />
      ))}
    </nav>
  );
}
