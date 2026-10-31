import {
  AppWindow,
  Brain,
  Container,
  Route as RouteIcon,
  Shield,
} from "lucide-react";
import MemoryIcon from "#/icons/memory_icon.svg?react";

export interface SettingsNavItem {
  icon: React.ReactElement;
  to: string;
  text: string;
  /** Short grey subline under the page title (`settings.tsx`). */
  subtitle: string;
}

export const OSS_NAV_ITEMS: SettingsNavItem[] = [
  {
    icon: <RouteIcon className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/meta-llm",
    text: "SETTINGS$NAV_META_LLM",
    subtitle: "SETTINGS$PAGE_META_LLM_SUBLINE",
  },
  {
    icon: <MemoryIcon width={16} height={16} />,
    to: "/settings/condenser",
    text: "SETTINGS$NAV_CONDENSER",
    subtitle: "SETTINGS$PAGE_CONDENSER_SUBLINE",
  },
  {
    // The agent's ``agent_context`` section, whatever the schema exposes in it
    // — today only persistent memory (``agent_context.load_memory``). Not
    // ``disabledByAcp``: the agent-server stamps the stored flag onto every
    // profile-resolved agent, ACP included.
    icon: <Brain className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/agent-context",
    text: "SETTINGS$NAV_AGENT_CONTEXT",
    subtitle: "SETTINGS$PAGE_AGENT_CONTEXT_SUBLINE",
  },
  {
    icon: <Shield className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/verification",
    text: "SETTINGS$NAV_VERIFICATION",
    subtitle: "SETTINGS$PAGE_VERIFICATION_SUBLINE",
  },
  {
    icon: <Container className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/runtime",
    text: "SETTINGS$NAV_RUNTIME",
    subtitle: "SETTINGS$PAGE_RUNTIME_SUBLINE",
  },
  {
    icon: <AppWindow className="size-4" strokeWidth={2} aria-hidden />,
    to: "/settings/app",
    text: "SETTINGS$NAV_APPLICATION",
    subtitle: "SETTINGS$PAGE_APPLICATION_SUBLINE",
  },
];

/**
 * The only OSS nav entry listed when the canvas is locked to a Cloud host —
 * i.e. deployed into SaaS / self-hosted OHE next to the OHE web app, whose own
 * settings shell (reached via "All Cloud Settings") owns everything else
 * (OHE-3168). The other pages are not reachable there either: the settings
 * layout redirects them here (OHE-3457).
 */
export const LOCKED_CLOUD_SETTINGS_NAV_PATH = "/settings/app";
