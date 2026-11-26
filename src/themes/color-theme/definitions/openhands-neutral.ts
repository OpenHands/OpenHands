import type { ColorThemeDefinition } from "../types";
import { NEUTRAL_SCALE, NEUTRAL_HEROUI } from "../palettes/neutral";

export const openhandsNeutral: ColorThemeDefinition = {
  label: "OpenHands-Neutral",
  appearance: "dark",
  scale: NEUTRAL_SCALE,
  tokens: {
    // Muted text clears 4.5:1 on every panel surface: 5.16:1 on the #313131
    // tertiary and 4.94:1 on the reported #343434 deep fill.
    "--oh-muted": "#A3A3A3",
  },
  // Each stop follows the same positional mapping as hero.ts:
  //   heroui-default-100 ← cool-grey-950 position ← neutral-950 (#181818)
  //   heroui-default-200 ← cool-grey-925 position ← neutral-900 (#202020)
  //   ...etc.
  heroui: NEUTRAL_HEROUI,
};
