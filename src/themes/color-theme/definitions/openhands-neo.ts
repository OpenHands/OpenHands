import type { AgentServerUICssVariableName } from "#/styles/agent-server-ui-style-scope";
import type { ColorThemeDefinition } from "../types";
import { NEUTRAL_SCALE, NEUTRAL_HEROUI } from "../palettes/neutral";

/** White button tokens plus the shared dark muted text value. */
const NEO_TOKENS: Partial<Record<AgentServerUICssVariableName, string>> = {
  "--oh-color-primary": "#ffffff",
  "--oh-accent": "#ffffff",
  "--oh-warning": "#ffffff",
  "--oh-muted": "#A3A3A3",
};

export const openhandsNeo: ColorThemeDefinition = {
  label: "OpenHands-Neo",
  appearance: "dark",
  scale: NEUTRAL_SCALE,
  heroui: NEUTRAL_HEROUI,
  tokens: NEO_TOKENS,
};
