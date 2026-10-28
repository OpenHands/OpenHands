import React from "react";
import { useTranslation } from "react-i18next";
import { useOptionalScrollContext } from "#/context/scroll-context";
import { useLoadOlderEvents } from "#/hooks/use-load-older-events";
import { I18nKey } from "#/i18n/declaration";
import { displayErrorToast } from "#/utils/custom-toast-handlers";

/** Hard guard against runaway page loads on huge histories (spec: 10 pages). */
const MAX_LOAD_OLDER_PAGES = 10;

/** Where scrollIntoView should land the target. */
const SCROLL_OPTIONS: ScrollIntoViewOptions = {
  block: "center",
  behavior: "smooth",
};

/** How long the highlight pulse stays on the target, ms. */
const HIGHLIGHT_MS = 1500;

/**
 * Jump to a message by its event id: scroll the `[data-event-id]` anchor to
 * the center of the chat scroll container and pulse-highlight it. If the
 * target is not in the loaded window, older pages are loaded until it
 * appears (bounded by MAX_LOAD_OLDER_PAGES); on exhaustion the localized
 * "not in loaded history" hint is surfaced instead of scrolling blindly.
 */
export function useJumpToEvent() {
  const { t } = useTranslation();
  const scrollContext = useOptionalScrollContext();
  const scrollRef = scrollContext?.scrollRef;
  // Load-older merges pages into the event store by conversation; passing no
  // id (undefined) matches the chat-surface default scope.
  const { loadOlder } = useLoadOlderEvents(undefined);

  const jump = React.useCallback(
    async (eventId: string) => {
      const container = scrollRef?.current;

      const findAnchor = () =>
        (container ?? document).querySelector<HTMLElement>(
          `[data-event-id="${CSS.escape(eventId)}"]`,
        );

      const reveal = (anchor: HTMLElement) => {
        anchor.scrollIntoView(SCROLL_OPTIONS);
        anchor.setAttribute("data-event-jump-highlight", "");
        window.setTimeout(() => {
          anchor.removeAttribute("data-event-jump-highlight");
        }, HIGHLIGHT_MS);
      };

      // The anchor may need several pages before it appears; re-query the
      // DOM after each loadOlder resolves.
      for (let pages = 0; ; pages += 1) {
        const anchor = findAnchor();
        if (anchor) {
          reveal(anchor);
          return;
        }
        if (pages >= MAX_LOAD_OLDER_PAGES) {
          displayErrorToast(t(I18nKey.CHAT_INTERFACE$REPLY_NOT_LOADED));
          return;
        }
        await loadOlder();
      }
    },
    [scrollRef, loadOlder, t],
  );

  return jump;
}
