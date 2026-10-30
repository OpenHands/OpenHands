/** Asks the conversation shell to reveal the automation setup agent panel. */
export const AUTOMATION_SETUP_SHOW_AGENT_EVENT =
  "openhands:automation-setup-show-agent";

export function requestAutomationSetupAgent() {
  window.dispatchEvent(new CustomEvent(AUTOMATION_SETUP_SHOW_AGENT_EVENT));
}
