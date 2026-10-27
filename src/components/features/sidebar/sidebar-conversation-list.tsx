import { getMarsBridge } from "#/api/mars/mars-tunnel-backend";
import { ConversationPanel } from "#/components/features/conversation-panel/conversation-panel";
import { useActiveBackend } from "#/contexts/active-backend-context";
import { MarsSessionPanel } from "./mars-session-panel";

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
  const { backend } = useActiveBackend();

  if (collapsed) {
    return null;
  }

  // A Managed Agents session's unit of work is the session, so its agent's
  // sessions replace the list, with this session's conversations nested.
  const showMarsSessions =
    Boolean(backend.marsSessionId) && getMarsBridge() !== null;

  return (
    <div className="flex flex-col flex-1 min-h-0">
      {/* Avoid overflow-hidden here: ConversationPanel's header uses `-ml-2.5` +
          `w-[calc(100%+0.625rem)]` to full-bleed the divider with `md:pr-0` on
          the aside; clipping would inset the border. Scroll stays on the inner
          list. */}
      <div className="flex min-h-0 w-full flex-1 flex-col">
        {showMarsSessions ? (
          <MarsSessionPanel backend={backend} />
        ) : (
          <ConversationPanel />
        )}
      </div>
    </div>
  );
}
