import { FiUsers, FiBriefcase } from "react-icons/fi";
import CreditCardIcon from "#/icons/credit-card.svg?react";
import KeyIcon from "#/icons/key.svg?react";
import LightbulbIcon from "#/icons/lightbulb.svg?react";
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
  // Org settings section (Admin/Owner only)
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
    to: "/settings",
    text: "COMMON$LANGUAGE_MODEL_LLM",
    section: "org",
  },
  // Personal settings section
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
    icon: <ServerProcessIcon width={22} height={22} />,
    to: "/settings/mcp",
    text: "SETTINGS$NAV_MCP",
    section: "personal",
  },
  // User settings section (no header shown)
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
  // Billing section (personal orgs only)
  {
    icon: <CreditCardIcon width={22} height={22} />,
    to: "/settings/billing",
    text: "SETTINGS$NAV_BILLING",
    section: "billing",
  },
  // Other items
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
  {
    icon: <CircuitIcon width={22} height={22} />,
    to: "/settings",
    text: "SETTINGS$NAV_LLM",
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
