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

const paper: ITheme = {
  black: "#29241f",
  red: "#9b4428",
  green: "#52644b",
  yellow: "#79601f",
  blue: "#3f5f7c",
  magenta: "#70456a",
  cyan: "#346458",
  white: "#6b6258",
  brightBlack: "#6b6258",
  brightRed: "#84371f",
  brightGreen: "#43563c",
  brightYellow: "#6b5219",
  brightBlue: "#304f6d",
  brightMagenta: "#60365a",
  brightCyan: "#28564a",
  brightWhite: "#29241f",
  selectionBackground: "rgba(155, 68, 40, 0.18)",
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
