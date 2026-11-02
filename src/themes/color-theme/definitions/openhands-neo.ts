import type { AgentServerUICssVariableName } from "#/styles/agent-server-ui-style-scope";
import type { ColorThemeDefinition } from "../types";
import { NEUTRAL_SCALE, NEUTRAL_HEROUI } from "../palettes/neutral";

/** White primary/accent tokens — used by OpenHands-Neo for button surfaces. */
const NEO_WHITE_BUTTON_TOKENS: Partial<
  Record<AgentServerUICssVariableName, string>
> = {
  "--oh-color-primary": "#ffffff",
  "--oh-accent": "#ffffff",
  "--oh-warning": "#ffffff",
};

export const openhandsNeo: ColorThemeDefinition = {
  label: "OpenHands-Neo",
  appearance: "dark",
  scale: NEUTRAL_SCALE,
  heroui: NEUTRAL_HEROUI,
  tokens: NEO_WHITE_BUTTON_TOKENS,
};
