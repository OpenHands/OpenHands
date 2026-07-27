import { AppWindow, Shield } from "lucide-react";
import KeyIcon from "#/icons/key.svg?react";
import MemoryIcon from "#/icons/memory_icon.svg?react";
import CircuitIcon from "#/icons/u-circuit.svg?react";
import RobotIcon from "#/icons/u-robot.svg?react";

export interface SettingsNavItem {
  icon: React.ReactElement;
  to: string;
  text: string;
  /** Short grey subline under the page title (`settings.tsx`). */
  subtitle: string;
}

export const OSS_NAV_ITEMS: SettingsNavItem[] = [
  {
    // "Agent" is the Agent Profile library: it lists the user's agent profiles
    // and its create/edit view is the reused Agent settings form plus a name.
    // The active profile is the current agent (#1571). Replaces the old split
    // of a global "Agent" form + a separate "Agent profiles" library.
    icon: <RobotIcon width={16} height={16} />,
    to: "/settings/agents",
    text: "SETTINGS$NAV_AGENT",
    subtitle: "SETTINGS$PAGE_AGENT_PROFILES_SUBLINE",
  },
  {
    icon: <CircuitIcon width={16} height={16} />,
    to: "/settings/llm",
    text: "SETTINGS$NAV_LLM",
    subtitle: "SETTINGS$PAGE_LLM_SUBLINE",
  },
  {
    icon: <MemoryIcon width={16} height={16} />,
    to: "/settings/condenser",
    text: "SETTINGS$NAV_CONDENSER",
    subtitle: "SETTINGS$PAGE_CONDENSER_SUBLINE",
  },
  {
    icon: <Shield className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/verification",
    text: "SETTINGS$NAV_VERIFICATION",
    subtitle: "SETTINGS$PAGE_VERIFICATION_SUBLINE",
  },
  {
    icon: <AppWindow className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/app",
    text: "SETTINGS$NAV_APPLICATION",
    subtitle: "SETTINGS$PAGE_APPLICATION_SUBLINE",
  },
  {
    icon: <KeyIcon width={16} height={16} />,
    to: "/settings/secrets",
    text: "SETTINGS$NAV_SECRETS",
    subtitle: "SETTINGS$PAGE_SECRETS_SUBLINE",
  },
];
