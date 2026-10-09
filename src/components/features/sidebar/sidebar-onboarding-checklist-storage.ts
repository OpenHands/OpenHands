import {
  SIDEBAR_ONBOARDING_CHECKLIST_CUSTOMIZE_EXPLORED_STORAGE_KEY,
  SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_CHANGE_EVENT,
  SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY,
  SIDEBAR_ONBOARDING_CHECKLIST_MINIMIZED_STORAGE_KEY,
  SIDEBAR_ONBOARDING_CHECKLIST_SLACK_JOINED_STORAGE_KEY,
} from "./sidebar-onboarding-checklist.constants";

export function readSidebarOnboardingChecklistDismissed(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return (
    window.localStorage.getItem(
      SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY,
    ) === "true"
  );
}

export function getSidebarOnboardingChecklistDismissedSnapshot(): boolean {
  return readSidebarOnboardingChecklistDismissed();
}

export function subscribeSidebarOnboardingChecklistDismissed(
  onStoreChange: () => void,
): () => void {
  const handleChange = () => onStoreChange();
  window.addEventListener("storage", handleChange);
  window.addEventListener(
    SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_CHANGE_EVENT,
    handleChange,
  );

  return () => {
    window.removeEventListener("storage", handleChange);
    window.removeEventListener(
      SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_CHANGE_EVENT,
      handleChange,
    );
  };
}

export function writeSidebarOnboardingChecklistDismissed(
  dismissed: boolean,
): void {
  window.localStorage.setItem(
    SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_STORAGE_KEY,
    dismissed ? "true" : "false",
  );
  window.dispatchEvent(
    new Event(SIDEBAR_ONBOARDING_CHECKLIST_DISMISSED_CHANGE_EVENT),
  );
}

export function readSidebarOnboardingChecklistMinimized(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return (
    window.localStorage.getItem(
      SIDEBAR_ONBOARDING_CHECKLIST_MINIMIZED_STORAGE_KEY,
    ) === "true"
  );
}

export function writeSidebarOnboardingChecklistMinimized(
  minimized: boolean,
): void {
  window.localStorage.setItem(
    SIDEBAR_ONBOARDING_CHECKLIST_MINIMIZED_STORAGE_KEY,
    minimized ? "true" : "false",
  );
}

export function readSidebarOnboardingChecklistCustomizeExplored(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return (
    window.localStorage.getItem(
      SIDEBAR_ONBOARDING_CHECKLIST_CUSTOMIZE_EXPLORED_STORAGE_KEY,
    ) === "true"
  );
}

export function writeSidebarOnboardingChecklistCustomizeExplored(
  explored: boolean,
): void {
  window.localStorage.setItem(
    SIDEBAR_ONBOARDING_CHECKLIST_CUSTOMIZE_EXPLORED_STORAGE_KEY,
    explored ? "true" : "false",
  );
}

export function readSidebarOnboardingChecklistSlackJoined(): boolean {
  if (typeof window === "undefined") {
    return false;
  }

  return (
    window.localStorage.getItem(
      SIDEBAR_ONBOARDING_CHECKLIST_SLACK_JOINED_STORAGE_KEY,
    ) === "true"
  );
}

export function writeSidebarOnboardingChecklistSlackJoined(
  joined: boolean,
): void {
  window.localStorage.setItem(
    SIDEBAR_ONBOARDING_CHECKLIST_SLACK_JOINED_STORAGE_KEY,
    joined ? "true" : "false",
  );
}
