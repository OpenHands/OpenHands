import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";

interface SidebarConversationListProps {
  /**
   * Whether the surrounding sidebar rail is rendering in its collapsed icon-
   * only variant. Passed from `SidebarRailBody` so the mobile drawer (which
   * renders an expanded rail regardless of the persisted desktop state) can
   * force this list back on.
   */
  collapsed: boolean;
}

/**
 * Conversation list section rendered inside the sidebar nav. The list itself
 * scrolls independently from the rest of the nav.
 *
 * In the collapsed sidebar variant the list reduces each row to a status
 * indicator + hover-preview.
 *
 * On desktop the aside uses `pr-0` so this list is full width to the rail;
 * nav links above keep their own horizontal padding.
 */
export function SidebarConversationList({
  collapsed,
}: SidebarConversationListProps) {
  if (collapsed) {
    return null;
  }

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Avoid overflow-hidden here: ConversationPanel's header uses `-ml-2.5` +
          `w-[calc(100%+0.625rem)]` to full-bleed the divider with `md:pr-0` on
          the aside; clipping would inset the border. Scroll stays on the inner
          list. */}
      <div className="flex min-h-0 w-full flex-1 flex-col">
        <ConversationPanel />
      </div>
    </div>
  );
}
