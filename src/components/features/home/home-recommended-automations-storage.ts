import {
  HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_CHANGE_EVENT,
  HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
} from "./home-recommended-automations.constants";

export function readHomeRecommendedAutomationsDismissed(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return (
    window.localStorage.getItem(
      HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
    ) === "true"
  );
}

export function getHomeRecommendedAutomationsDismissedSnapshot(): boolean {
  return readHomeRecommendedAutomationsDismissed();
}

export function subscribeHomeRecommendedAutomationsDismissed(
  onStoreChange: () => void,
): () => void {
  const handleChange = () => onStoreChange();
  window.addEventListener("storage", handleChange);
  window.addEventListener(
    HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_CHANGE_EVENT,
    handleChange,
  );

  return () => {
    window.removeEventListener("storage", handleChange);
    window.removeEventListener(
      HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_CHANGE_EVENT,
      handleChange,
    );
  };
}

export function writeHomeRecommendedAutomationsDismissed(
  dismissed: boolean,
): void {
  window.localStorage.setItem(
    HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_STORAGE_KEY,
    dismissed ? "true" : "false",
  );
  window.dispatchEvent(
    new Event(HOME_RECOMMENDED_AUTOMATIONS_DISMISSED_CHANGE_EVENT),
  );
}
