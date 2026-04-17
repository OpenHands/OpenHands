import { FiUsers, FiBriefcase } from "react-icons/fi";
import CreditCardIcon from "#/icons/credit-card.svg?react";
import KeyIcon from "#/icons/key.svg?react";
import LightbulbIcon from "#/icons/lightbulb.svg?react";
import LockIcon from "#/icons/lock.svg?react";
import MemoryIcon from "#/icons/memory_icon.svg?react";
import ServerProcessIcon from "#/icons/server-process.svg?react";
import SettingsGearIcon from "#/icons/settings-gear.svg?react";
import CircuitIcon from "#/icons/u-circuit.svg?react";
import PuzzlePieceIcon from "#/icons/u-puzzle-piece.svg?react";
import UserIcon from "#/icons/user.svg?react";

export type SettingsNavSection =
  | "org"
  | "personal"
  | "user"
  | "billing"
  | "other";

export interface SettingsNavItem {
  icon: React.ReactElement;
  to: string;
  text: string;
  section?: SettingsNavSection;
}

export const SAAS_NAV_ITEMS: SettingsNavItem[] = [
  {
    icon: <FiBriefcase size={22} />,
    to: "/settings/org",
    text: "SETTINGS$NAV_ORGANIZATION",
    section: "org",
  },
  {
    icon: <FiUsers size={22} />,
    to: "/settings/org-members",
    text: "SETTINGS$NAV_ORG_MEMBERS",
    section: "org",
  },
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings/org-defaults",
    text: "COMMON$LANGUAGE_MODEL_LLM",
    section: "org",
  },
  {
    icon: <MemoryIcon width={22} height={22} />,
    to: "/settings/org-defaults/condenser",
    text: "SETTINGS$NAV_CONDENSER",
    section: "org",
  },
  {
    icon: <LockIcon width={22} height={22} />,
    to: "/settings/org-defaults/verification",
    text: "SETTINGS$NAV_VERIFICATION",
    section: "org",
  },
  // Agent Type is the top-level selector — defaults to OpenHands (the
  // standard LLM agent). Saving ACP here reshapes the nav below so the
  // user only sees settings that apply to their chosen agent type.
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings/agent-type",
    text: "SETTINGS$NAV_AGENT_TYPE",
    section: "personal",
  },
  // ACP-agent sub-configuration. Only visible when agent_kind=acp —
  // see ``ACP_ONLY_NAV_PATHS`` in ``use-settings-nav-items.ts``.
  {
    icon: <ServerProcessIcon width={22} height={22} />,
    to: "/settings/acp-server",
    text: "SETTINGS$NAV_ACP_SERVER",
    section: "personal",
  },
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings/acp-model",
    text: "SETTINGS$NAV_ACP_MODEL",
    section: "personal",
  },
  // OpenHands-agent sub-configuration. These nav entries hide when
  // agent_kind=acp because the ACP subprocess manages its own
  // LLM/condenser/security/MCP — see ``LLM_ONLY_NAV_PATHS`` in
  // ``use-settings-nav-items.ts``.
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings",
    text: "COMMON$LANGUAGE_MODEL_LLM",
    section: "personal",
  },
  {
    icon: <MemoryIcon width={22} height={22} />,
    to: "/settings/condenser",
    text: "SETTINGS$NAV_CONDENSER",
    section: "personal",
  },
  {
    icon: <LockIcon width={22} height={22} />,
    to: "/settings/verification",
    text: "SETTINGS$NAV_VERIFICATION",
    section: "personal",
  },
  {
    icon: <ServerProcessIcon width={22} height={22} />,
    to: "/settings/mcp",
    text: "SETTINGS$NAV_MCP",
    section: "personal",
  },
  {
    icon: <KeyIcon width={22} height={22} />,
    to: "/settings/api-keys",
    text: "SETTINGS$NAV_API_KEYS",
    section: "personal",
  },
  {
    icon: <KeyIcon width={22} height={22} />,
    to: "/settings/secrets",
    text: "SETTINGS$NAV_SECRETS",
    section: "personal",
  },
  {
    icon: <UserIcon width={22} height={22} />,
    to: "/settings/user",
    text: "SETTINGS$NAV_USER",
    section: "user",
  },
  {
    icon: <SettingsGearIcon width={22} height={22} />,
    to: "/settings/app",
    text: "SETTINGS$NAV_APPLICATION",
    section: "user",
  },
  {
    icon: <CreditCardIcon width={22} height={22} />,
    to: "/settings/billing",
    text: "SETTINGS$NAV_BILLING",
    section: "billing",
  },
  {
    icon: <PuzzlePieceIcon width={22} height={22} />,
    to: "/settings/integrations",
    text: "SETTINGS$NAV_INTEGRATIONS",
    section: "other",
  },
  {
    icon: <LightbulbIcon width={22} height={22} />,
    to: "/settings/skills",
    text: "SETTINGS$NAV_SKILLS",
    section: "other",
  },
];

export const OSS_NAV_ITEMS: SettingsNavItem[] = [
  // Agent Type — top-level selector (OpenHands / ACP).
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings/agent-type",
    text: "SETTINGS$NAV_AGENT_TYPE",
  },
  // ACP-agent sub-configuration (only visible in ACP mode).
  {
    icon: <ServerProcessIcon width={22} height={22} />,
    to: "/settings/acp-server",
    text: "SETTINGS$NAV_ACP_SERVER",
  },
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings/acp-model",
    text: "SETTINGS$NAV_ACP_MODEL",
  },
  // OpenHands-agent sub-configuration (all hidden in ACP mode).
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings",
    text: "SETTINGS$NAV_LLM",
  },
  {
    icon: <MemoryIcon width={22} height={22} />,
    to: "/settings/condenser",
    text: "SETTINGS$NAV_CONDENSER",
  },
  {
    icon: <LockIcon width={22} height={22} />,
    to: "/settings/verification",
    text: "SETTINGS$NAV_VERIFICATION",
  },
  {
    icon: <ServerProcessIcon width={22} height={22} />,
    to: "/settings/mcp",
    text: "SETTINGS$NAV_MCP",
  },
  {
    icon: <LightbulbIcon width={22} height={22} />,
    to: "/settings/skills",
    text: "SETTINGS$NAV_SKILLS",
  },
  {
    icon: <PuzzlePieceIcon width={22} height={22} />,
    to: "/settings/integrations",
    text: "SETTINGS$NAV_INTEGRATIONS",
  },
  {
    icon: <SettingsGearIcon width={22} height={22} />,
    to: "/settings/app",
    text: "SETTINGS$NAV_APPLICATION",
  },
  {
    icon: <KeyIcon width={22} height={22} />,
    to: "/settings/secrets",
    text: "SETTINGS$NAV_SECRETS",
  },
];
