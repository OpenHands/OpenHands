import { createContext, useContext } from "react";

/**
 * True when the consuming event message is rendered inside an `EventGroup`'s
 * expanded children. Components like `GenericEventMessage` use this to drop
 * their own left bar so we don't render a redundant inner bar on top of the
 * group's outer bar.
 */
export const IsInEventGroupContext = createContext(false);

/**
 * Spacing for an event row that stands alone in the transcript rather than
 * inside a group: the same margins as `EventGroup`, including the trimmed top
 * margin of agent text that follows, so single rows and groups line up.
 */
const STANDALONE_EVENT_ROW_CLASS = "mt-2 mb-0 [&+article]:mt-2";

export const useStandaloneEventRowClass = (): string | undefined =>
  useContext(IsInEventGroupContext) ? undefined : STANDALONE_EVENT_ROW_CLASS;
