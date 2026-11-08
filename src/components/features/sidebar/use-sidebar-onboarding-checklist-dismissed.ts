import { useSyncExternalStore } from "react";
import {
  getSidebarOnboardingChecklistDismissedSnapshot,
  subscribeSidebarOnboardingChecklistDismissed,
} from "./sidebar-onboarding-checklist-storage";

export function useSidebarOnboardingChecklistDismissed(): boolean {
  return useSyncExternalStore(
    subscribeSidebarOnboardingChecklistDismissed,
    getSidebarOnboardingChecklistDismissedSnapshot,
    () => false,
  );
}
