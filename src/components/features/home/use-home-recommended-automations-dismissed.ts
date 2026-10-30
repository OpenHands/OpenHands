import { useSyncExternalStore } from "react";
import {
  getHomeRecommendedAutomationsDismissedSnapshot,
  subscribeHomeRecommendedAutomationsDismissed,
} from "./home-recommended-automations-storage";

export function useHomeRecommendedAutomationsDismissed(): boolean {
  return useSyncExternalStore(
    subscribeHomeRecommendedAutomationsDismissed,
    getHomeRecommendedAutomationsDismissedSnapshot,
    () => false,
  );
}
