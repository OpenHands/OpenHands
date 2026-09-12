import type { ColorThemeDefinition } from "../types";
import { NEUTRAL_SCALE, NEUTRAL_HEROUI } from "../palettes/neutral";

export const openhandsNeutral: ColorThemeDefinition = {
  label: "OpenHands-Neutral",
  appearance: "dark",
  scale: NEUTRAL_SCALE,
  // Each stop follows the same positional mapping as hero.ts:
  //   heroui-default-100 ← cool-grey-950 position ← neutral-950 (#181818)
  //   heroui-default-200 ← cool-grey-925 position ← neutral-900 (#202020)
  //   ...etc.
  heroui: NEUTRAL_HEROUI,
};
