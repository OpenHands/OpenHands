import { create } from "zustand";
import { devtools } from "zustand/middleware";

interface SkillInstallBannerState {
  /**
   * Install observation event ids the user dismissed. Session-only by
   * design: the "skill not active in this conversation" condition persists
   * across reloads, so the banner truthfully reappearing then is fine. A
   * new install (new event id) resurfaces the banner after a dismissal;
   * replays of the same event stay dismissed.
   */
  dismissedEventIds: Record<string, true>;
}

interface SkillInstallBannerActions {
  dismiss: (eventIds: string[]) => void;
}

type SkillInstallBannerStore = SkillInstallBannerState &
  SkillInstallBannerActions;

const initialState: SkillInstallBannerState = { dismissedEventIds: {} };

export const useSkillInstallBannerStore = create<SkillInstallBannerStore>()(
  devtools(
    (set) => ({
      ...initialState,
      dismiss: (eventIds) =>
        set((s) => ({
          dismissedEventIds: {
            ...s.dismissedEventIds,
            ...Object.fromEntries(eventIds.map((id) => [id, true as const])),
          },
        })),
    }),
    { name: "SkillInstallBannerStore" },
  ),
);
