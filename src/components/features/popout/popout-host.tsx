import React from "react";
import { useNavigation } from "#/context/navigation-context";
import {
  POPOUT_DOCK_INSET_PX,
  POPOUT_EXPANDED_WIDTH_PX,
  POPOUT_GAP_PX,
  POPOUT_MINIMIZED_WIDTH_PX,
  POPOUT_OVERFLOW_BUTTON_SIZE_PX,
  POPOUT_Z_INDEX,
  usePopoutStore,
} from "#/stores/popout-store";
import { layoutPopouts } from "./popout-layout";
import { PopoutChrome } from "./popout-chrome";
import { PopoutConversation } from "./popout-conversation";
import { PopoutOverflowMenu } from "./popout-overflow-menu";
import { PopoutStatusDot } from "./popout-status-dot";

function useViewportWidth(): number {
  const [width, setWidth] = React.useState(() =>
    typeof window === "undefined" ? 1280 : window.innerWidth,
  );

  React.useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

  return width;
}

/**
 * Bottom-right dock of floating popouts. Renders above page content and
 * below modals/dropdowns. Multiple popouts sit side-by-side; overflowed,
 * least-recently-used entries move into a count selector at the far right.
 */
export function PopoutHost() {
  const popouts = usePopoutStore((state) => state.popouts);
  const closePopout = usePopoutStore((state) => state.closePopout);
  const toggleMinimized = usePopoutStore((state) => state.toggleMinimized);
  const expandPopout = usePopoutStore((state) => state.expandPopout);
  const { navigate } = useNavigation();
  const viewportWidth = useViewportWidth();

  const layout = React.useMemo(
    () => layoutPopouts(popouts, viewportWidth),
    [popouts, viewportWidth],
  );

  if (layout.visible.length === 0 && layout.hidden.length === 0) {
    return null;
  }

  const availablePopoutWidth =
    viewportWidth -
    POPOUT_DOCK_INSET_PX * 2 -
    (layout.hidden.length > 0
      ? POPOUT_OVERFLOW_BUTTON_SIZE_PX + POPOUT_GAP_PX
      : 0);

  return (
    <div
      data-testid="popout-host"
      className="pointer-events-none fixed inset-x-0 bottom-0 flex items-end justify-end"
      style={{
        zIndex: POPOUT_Z_INDEX,
        paddingRight: POPOUT_DOCK_INSET_PX,
        paddingBottom: POPOUT_DOCK_INSET_PX,
        gap: POPOUT_GAP_PX,
      }}
    >
      {layout.visible.map((entry) => {
        const preferredWidth =
          entry.mode === "expanded"
            ? POPOUT_EXPANDED_WIDTH_PX
            : POPOUT_MINIMIZED_WIDTH_PX;

        const handleMaximize = () => {
          closePopout(entry.conversationId);
          navigate(`/conversations/${entry.conversationId}`);
        };

        const handleToggleMinimized = () => {
          if (entry.mode === "minimized") {
            expandPopout(entry.conversationId);
          } else {
            toggleMinimized(entry.conversationId);
          }
        };

        return (
          <div
            key={entry.conversationId}
            data-testid={`popout-${entry.conversationId}`}
            className="pointer-events-auto min-w-0 shrink-0"
            style={{
              width: preferredWidth,
              maxWidth: Math.max(0, availablePopoutWidth),
            }}
          >
            <PopoutChrome
              title={entry.title}
              statusIndicator={
                <PopoutStatusDot conversationId={entry.conversationId} />
              }
              mode={entry.mode}
              onMaximize={handleMaximize}
              onToggleMinimized={handleToggleMinimized}
              onClose={() => closePopout(entry.conversationId)}
            >
              <PopoutConversation conversationId={entry.conversationId} />
            </PopoutChrome>
          </div>
        );
      })}
      {layout.hidden.length > 0 ? (
        <div className="pointer-events-auto">
          <PopoutOverflowMenu
            hiddenPopouts={layout.hidden}
            onSelect={expandPopout}
          />
        </div>
      ) : null}
    </div>
  );
}
