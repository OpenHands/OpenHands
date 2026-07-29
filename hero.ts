import { heroui } from "@heroui/react";

export default heroui({
  defaultTheme: "dark",
  layout: {
    radius: {
      small: "5px",
      large: "20px",
    },
  },
  themes: {
    dark: {
      colors: {
        primary: "#4465DB",

        // Map HeroUI's zinc-based semantic colours to our cool-grey palette.
        // This ensures every HeroUI component that uses bg-default, bg-content*,
        // etc. stays within the same colour family as the rest of the UI.

        background: {
          DEFAULT: "#0B0E14", // cool-grey-950 — app shell base
          foreground: "#F7F9FC", // cool-grey-50
        },

        foreground: {
          DEFAULT: "#C3CDDC", // cool-grey-300 — primary readable text
          "50": "#05070A", // cool-grey-975
          "100": "#0B0E14", // cool-grey-950
          "200": "#21252F", // cool-grey-925
          "300": "#2C313F", // cool-grey-900
          "400": "#383F50", // cool-grey-800
          "500": "#4B5468", // cool-grey-700
          "600": "#626D82", // cool-grey-600
          "700": "#7E8A9E", // cool-grey-500
          "800": "#A3B0C4", // cool-grey-400
          "900": "#C3CDDC", // cool-grey-300
        },

        // Surface layers: panel → card → inner card → inset
        content1: { DEFAULT: "#21252F", foreground: "#EEF2F7" }, // cool-grey-925 / 100
        content2: { DEFAULT: "#2C313F", foreground: "#DCE3EE" }, // cool-grey-900 / 200
        content3: { DEFAULT: "#383F50", foreground: "#C3CDDC" }, // cool-grey-800 / 300
        content4: { DEFAULT: "#4B5468", foreground: "#A3B0C4" }, // cool-grey-700 / 400

        focus: {
          DEFAULT: "#ffffff", // white focus ring — visible on all dark surfaces
        },
        default: {
          "50": "#05070A", // cool-grey-975
          "100": "#0B0E14", // cool-grey-950
          "200": "#21252F", // cool-grey-925
          "300": "#2C313F", // cool-grey-900
          "400": "#383F50", // cool-grey-800
          "500": "#4B5468", // cool-grey-700
          "600": "#626D82", // cool-grey-600
          "700": "#7E8A9E", // cool-grey-500
          "800": "#A3B0C4", // cool-grey-400
          "900": "#C3CDDC", // cool-grey-300
          DEFAULT: "#383F50", // cool-grey-800 — hover/selected tint
          foreground: "#F7F9FC", // cool-grey-50 — text on default bg
        },
      },
    },
  },
});
