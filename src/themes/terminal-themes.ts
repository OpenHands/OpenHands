import type { ITheme } from "@xterm/xterm";
import { COLOR_THEMES, type ColorThemeKey } from "./color-themes";

// Specify every ANSI slot so switching back to dark also resets light colors.
const dark: ITheme = {
  black: "#000000",
  red: "#cd0000",
  green: "#00cd00",
  yellow: "#cdcd00",
  blue: "#0000ee",
  magenta: "#cd00cd",
  cyan: "#00cdcd",
  white: "#e5e5e5",
  brightBlack: "#7f7f7f",
  brightRed: "#ff0000",
  brightGreen: "#00ff00",
  brightYellow: "#ffff00",
  brightBlue: "#5c5cff",
  brightMagenta: "#ff00ff",
  brightCyan: "#00ffff",
  brightWhite: "#ffffff",
};

const light: ITheme = {
  black: "#1f1f1f",
  red: "#a31515",
  green: "#167040",
  yellow: "#795e00",
  blue: "#0451a5",
  magenta: "#800080",
  cyan: "#006b73",
  white: "#616161",
  brightBlack: "#616161",
  brightRed: "#b52020",
  brightGreen: "#12743c",
  brightYellow: "#806000",
  brightBlue: "#005cc5",
  brightMagenta: "#950095",
  brightCyan: "#007880",
  brightWhite: "#424242",
};

const solarized: ITheme = {
  black: "#073642",
  red: "#b52624",
  green: "#586e00",
  yellow: "#826200",
  blue: "#006ab1",
  magenta: "#a72d70",
  cyan: "#147a74",
  white: "#586e75",
  brightBlack: "#657b83",
  brightRed: "#b84313",
  brightGreen: "#657b00",
  brightYellow: "#936f00",
  brightBlue: "#0879ba",
  brightMagenta: "#6c71c4",
  brightCyan: "#16827a",
  brightWhite: "#002b36",
};

// Every ANSI slot is an ink that stays legible on newsprint; "white" is the
// faded ink, and bright variants print one shade heavier.
const paper: ITheme = {
  black: "#1d1c19",
  red: "#a01f1b",
  green: "#3a5636",
  yellow: "#6a4f0e",
  blue: "#21436b",
  magenta: "#563875",
  cyan: "#1f5552",
  white: "#585752",
  brightBlack: "#585752",
  brightRed: "#861814",
  brightGreen: "#2f472c",
  brightYellow: "#594209",
  brightBlue: "#193555",
  brightMagenta: "#472c62",
  brightCyan: "#174442",
  brightWhite: "#1d1c19",
  selectionBackground: "rgba(33, 67, 107, 0.2)",
};

const palettes: Record<ColorThemeKey, ITheme> = {
  "openhands-deepsea": dark,
  "openhands-neutral": dark,
  "openhands-neo": dark,
  "light-plus": light,
  "solarized-light": solarized,
  paper,
};

export function getTerminalTheme(
  key: ColorThemeKey,
  foreground: string,
): ITheme {
  return {
    ...palettes[key],
    background: "rgba(0, 0, 0, 0)",
    foreground,
    cursor: foreground,
    selectionBackground:
      palettes[key].selectionBackground ??
      (COLOR_THEMES[key].appearance === "light"
        ? "rgba(0, 80, 160, 0.2)"
        : "rgba(255, 255, 255, 0.25)"),
  };
}
